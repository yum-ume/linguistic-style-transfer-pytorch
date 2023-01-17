import torch
import torch.nn as nn
from linguistic_style_transfer_pytorch.config import ModelConfig, GeneralConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

mconfig = ModelConfig()
gconfig = GeneralConfig()


class AdversarialVAE(nn.Module):
    """
    Model architecture defined according to the paper
    'Disentangled Representation Learning for Non-Parallel Text Style Transfer'
    https://www.aclweb.org/anthology/P19-1041.pdf

    """

    def __init__(self, weight):
        """
        Initialize networks
        """
        super(AdversarialVAE, self).__init__()
        # word embeddings
        if mconfig.use_prepro_embed:
            self.encoder_embedding = nn.Embedding.from_pretrained(weight)
            self.decoder_embedding = nn.Embedding.from_pretrained(weight)
        else:
            self.encoder_embedding = nn.Embedding(mconfig.vocab_size, mconfig.embedding_size)
            self.decoder_embedding = nn.Embedding(mconfig.vocab_size, mconfig.embedding_size)

        #================ Encoder model =============#
        self.encoder = nn.GRU(
            mconfig.embedding_size, mconfig.hidden_dim, batch_first=True, bidirectional=True)
        # content latent embedding
        self.content_mu = nn.Linear(
            2*mconfig.hidden_dim, mconfig.content_hidden_dim)
        self.content_log_var = nn.Linear(
            2*mconfig.hidden_dim, mconfig.content_hidden_dim)
        # style latent embedding
        self.style_mu = nn.Linear(
            2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        self.style_log_var = nn.Linear(
            2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        #=============== Discriminator/adversary============#
        self.style_disc = nn.Sequential(
            nn.Linear(mconfig.content_hidden_dim, mconfig.content_hidden_dim),
            nn.LeakyReLU()
            )
        self.style_disc_pred = nn.Linear(
            mconfig.content_hidden_dim, mconfig.num_style)

        self.content_disc = nn.Sequential(
            nn.Linear(mconfig.style_hidden_dim, mconfig.content_bow_dim),
            nn.LeakyReLU()
            )
        self.content_disc_pred = nn.Linear(
            mconfig.content_bow_dim, mconfig.content_bow_dim)
        #=============== Classifier =============#
        self.content_classifier = nn.Sequential(
            nn.Linear(mconfig.content_hidden_dim, mconfig.content_bow_dim),
            nn.LeakyReLU()
            )
        self.style_classifier = nn.Sequential(
            nn.Linear(mconfig.style_hidden_dim, mconfig.num_style),
            nn.LeakyReLU()
            )
        #=============== Decoder =================#
        self.dense = nn.Sequential(
            nn.Linear(mconfig.style_hidden_dim + mconfig.content_hidden_dim, mconfig.hidden_dim),
            nn.LeakyReLU()
            )
        # Note: input embeddings are concatenated with the sampled latent vector at every step
        self.decoder = nn.GRUCell(
            mconfig.embedding_size + mconfig.hidden_dim, mconfig.hidden_dim)
        self.projector = nn.Linear(mconfig.hidden_dim, mconfig.vocab_size)
        #============== Average label embedding ======#
        # concat all label embedding per label type, then get means
        # mean cal is at train.py because this model works per batch
        # here we just correct all label embedding 
        self.style_embedding_map = dict()
        # dropout
        self.dropout = lambda x: nn.functional.dropout(x, p=mconfig.dropout)

    def forward(self, sequences, seq_lengths, style_labels, content_bow, iteration, last_epoch):
        """
        Args:
            sequences : token indices of input sentences of shape = (batch_size,max_seq_length)
            seq_lengths: actual lengths of input sentences before padding, shape = (batch_size,1)
            style_labels: labels of sentiment of the input sentences, shape = (batch_size,2)
            content_bow: Bag of Words representations of the input sentences, shape = (batch_size,bow_hidden_size)
            iteration: number of iterations completed till now; used for KL annealing
            last_epoch: save average style embeddings if last_epoch is true

        Returns:
            content_disc_loss: loss incurred by content discriminator/adversary
            style_disc_loss  : loss incurred by style discriminator/adversary
            vae_and_classifier_loss : consists of loss incurred by autoencoder, content and style
                                      classifiers
        """
        # pack the sequences to reduce unnecessary computations
        # It requires the sentences to be sorted in descending order to take
        # full advantage
        embedded_seqs = self.dropout(self.encoder_embedding(sequences))
        packed_seqs = pack_padded_sequence(
            embedded_seqs, lengths=seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, h_n = self.encoder(packed_seqs)
        sentence_emb = h_n.transpose(0, 1).reshape(embedded_seqs.size(0), -1)
        
        # get content and style embeddings from the sentence embeddings,i.e. final_hidden_state
        content_emb_mu, content_emb_log_var = self.get_content_emb(
            sentence_emb)
        style_emb_mu, style_emb_log_var = self.get_style_emb(
            sentence_emb)
        # sample content and style embeddings from their respective latent spaces
        sampled_content_emb = self.sample_prior(
            content_emb_mu, content_emb_log_var)
        sampled_style_emb = self.sample_prior(
            style_emb_mu, style_emb_log_var)
        
        # Update the average style embeddings for different styles
        # This will be used in transfering the style of a sentence
        # during inference
        if last_epoch:
            self.get_average_label_emb(style_emb_mu, style_labels)

        #=========== Losses on content space =============#
        # Discriminator Loss
        content_adv_preds = self.get_content_disc_preds(style_emb_mu.detach())
        content_adv_loss = self.get_content_disc_loss(
            content_adv_preds, content_bow)
        # adversarial entropy
        content_disc_preds = nn.Softmax(dim=1)(self.get_content_disc_preds(style_emb_mu))
        content_entropy_loss = self.get_entropy_loss(content_disc_preds)
        # Multitask loss
        content_mul_loss = self.get_content_mul_loss(
            content_emb_mu, content_bow)

        #============ Losses on style space ================#
        # Discriminator loss
        style_adv_preds = self.get_style_disc_preds(content_emb_mu.detach())
        style_adv_loss = self.get_style_disc_loss(
            style_adv_preds, style_labels)
        # adversarial entropy
        style_disc_preds = nn.Softmax(dim=1)(self.get_style_disc_preds(content_emb_mu))
        style_entropy_loss = self.get_entropy_loss(style_disc_preds)
        # Multitask loss
        style_mul_loss = self.get_style_mul_loss(
            style_emb_mu, style_labels)

        #============== KL losses ===========#
        style_kl_weight, content_kl_weight = 0, 0
        # Style space
        unweighted_style_kl_loss = self.get_kl_loss(
            style_emb_mu, style_emb_log_var)
        if iteration < mconfig.kl_anneal_iterations:
            style_kl_weight = self.get_annealed_weight(
                iteration, mconfig.style_kl_lambda)
        style_kl_loss = unweighted_style_kl_loss * style_kl_weight
        
        # Content space
        unweighted_content_kl_loss = self.get_kl_loss(
            content_emb_mu, content_emb_log_var)
        if iteration < mconfig.kl_anneal_iterations:
            content_kl_weight = self.get_annealed_weight(
                iteration, mconfig.content_kl_lambda)
        content_kl_loss = unweighted_content_kl_loss * content_kl_weight

        #=============== reconstruction ================#
        # Generative embedding
        generative_emb = self.dense(torch.cat(
            (style_emb_mu, content_emb_mu), axis=1))
        reconstructed_sentences = self.generate_sentences(
            sequences, generative_emb)
        reconstruction_loss = self.get_recon_loss(
            reconstructed_sentences, sequences)
        #================ total weighted loss ==========#
        vae_and_classifier_loss = reconstruction_loss \
                                  + content_kl_loss \
                                  + style_kl_loss \
                                  + mconfig.content_multitask_loss_weight * content_mul_loss \
                                  - mconfig.content_adversary_loss_weight * content_entropy_loss \
                                  + mconfig.style_multitask_loss_weight * style_mul_loss \
                                  - mconfig.style_adversary_loss_weight * style_entropy_loss

        return content_adv_loss, style_adv_loss, vae_and_classifier_loss

    def get_style_content_emb(self, sequences, seq_lengths, style_labels, content_bow):
        """
        Args:
            sequences : token indices of input sentences of shape = (batch_size,max_seq_length)
            seq_lengths: actual lengths of input sentences before padding, shape = (batch_size,1)
            style_labels: labels of sentiment of the input sentences, shape = (batch_size,2)
            content_bow: Bag of Words representations of the input sentences, shape = (batch_size,bow_hidden_size)
        Returns:
            sampled_content_emb: content embeddings sampled from the content latent space, shape=(batch_size,content_hid_dim)
            sampled_style_emb:   style embeddings sampled from the style latent space, shape=(batch_size,style_hid_dim)

        """
        with torch.no_grad():
            # pack the sequences to reduce unnecessary computations
            # It requires the sentences to be sorted in descending order to take
            # full advantage
            seq_lengths, perm_index = seq_lengths.sort(descending=True)
            sequences = sequences[perm_index]
            embedded_seqs = self.embedding(sequences)
            packed_seqs = pack_padded_sequence(
                embedded_seqs, lengths=seq_lengths, batch_first=True)
            packed_output, (_) = self.encoder(packed_seqs)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            sentence_emb = output[torch.arange(output.size(0)), seq_lengths-1]
            # get content and style embeddings from the sentence embeddings,i.e. final_hidden_state
            content_emb_mu, content_emb_log_var = self.get_content_emb(
                sentence_emb)
            style_emb_mu, style_emb_log_var = self.get_style_emb(
                sentence_emb)
            # sample content and style embeddings from their respective latent spaces
            sampled_content_emb = self.sample_prior(
                content_emb_mu, content_emb_log_var)
            sampled_style_emb = self.sample_prior(
                style_emb_mu, style_emb_log_var)

        return sampled_content_emb, sampled_style_emb

    def get_params(self):
        """
        Returns:
            content_disc_params: parameters of the content discriminator/adversary
            style_disc_params  : parameters of the style discriminator/adversary
            other_params       : parameters of the vae and classifiers
        """

        content_disc_params = list(self.content_disc.parameters()) + list(self.content_disc_pred.parameters())
        style_disc_params = list(self.style_disc.parameters()) + list(self.style_disc_pred.parameters())
        other_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) \
                       + list(self.dense.parameters()) \
                       + list(self.encoder_embedding.parameters()) \
                       + list(self.decoder_embedding.parameters()) \
                       + list(self.projector.parameters()) \
                       + list(self.content_mu.parameters()) + list(self.content_log_var.parameters()) \
                       + list(self.style_mu.parameters())+list(self.style_log_var.parameters()) \
                       + list(self.content_classifier.parameters()) \
                       + list(self.style_classifier.parameters()) \
                       + list(self.content_disc.parameters()) + list(self.content_disc_pred.parameters()) \
                       + list(self.style_disc.parameters()) + list(self.style_disc_pred.parameters())

        return content_disc_params, style_disc_params, other_params

    def get_content_emb(self, sentence_emb):
        """
        Args:
            sentence_emb: sentence embeddings of all the sentences in the batch, shape=(batch_size,2*gru_hidden_dim)
        Returns:
            mu: embedding of the mean of the Gaussian distribution of the content's latent space
            log_var: embedding of the log of variance of the Gaussian distribution of the content's latent space
        """
        mu = self.content_mu(sentence_emb)
        log_var = self.content_log_var(sentence_emb)

        return mu, log_var

    def get_style_emb(self, sentence_emb):
        """
        Args:
            sentence_emb: sentence embeddings of all the sentences in the batch, shape=(batch_size,2*gru_hidden_dim)
        Returns:
            mu: embedding of the mean of the Gaussian distribution of the style's latent space
            log_var: embedding of the log of variance of the Gaussian distribution of the style's latent space
        """
        mu = self.style_mu(sentence_emb)
        log_var = self.style_log_var(sentence_emb)

        return mu, log_var

    def sample_prior(self, mu, log_var):
        """
        Returns samples drawn from the latent space constrained to
        follow diagonal Gaussian
        """
        epsilon = torch.randn(mu.size(1),device=mu.device)
        return mu + epsilon*torch.exp(log_var)

    def get_average_label_emb(self, style_emb, style_labels):
        """
        Args:
            style_emb: batch of sampled style embeddings of the input sentences,shape = (batch_size,mconfig.style_hidden_dim)
            style_labels: style labels of the corresponding input sentences,shape = (batch_size,2)
        """
        for idx in range(len(style_labels)):
            label = style_labels[idx].tolist().index(1)
            if label not in self.style_embedding_map:
                self.style_embedding_map[label] = list()
            self.style_embedding_map[label].append(style_emb[idx].unsqueeze(0))
            
    def get_content_disc_preds(self, style_emb):
        """
        Returns predictions about the content using style embedding
        as input
        output shape : [batch_size,content_bow_dim]
        """
        # predictions
        # Note: detach the style embedding since when don't want the gradient to flow
        #       all the way to the encoder. content_disc_loss is used only to change the
        #       parameters of the discriminator network
        content_mlp = self.content_disc(self.dropout(style_emb))
        content_preds = self.content_disc_pred(content_mlp)
        
        return content_preds

    def get_content_disc_loss(self, content_disc_preds, content_bow):
        """
        Returns:
        cross entropy loss of content discriminator
        """
        # calculate cross entropy loss
        content_disc_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(content_disc_preds, content_bow)

        return content_disc_loss

    def get_style_disc_preds(self, content_emb):
        """
        Returns predictions about style using content embeddings
        as input
        output shape: [batch_size,num_style]
        """
        # predictions
        # Note: detach the content embedding since when don't want the gradient to flow
        #       all the way to the encoder. style_disc_loss is used only to change the
        #       parameters of the discriminator network
        style_mlp =self.style_disc(self.dropout(content_emb))
        style_preds = self.style_disc_pred(style_mlp)

        return style_preds

    def get_style_disc_loss(self, style_disc_preds, style_labels):
        """
        Returns:
        cross entropy loss of style discriminator
        """
        # calculate cross entropy loss
        style_disc_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(style_disc_preds, style_labels)

        return style_disc_loss

    def get_entropy_loss(self, preds):
        """
        Returns the entropy loss: negative of the entropy present in the
        input distribution
        """
        return torch.mean(torch.sum(preds * torch.log(preds + mconfig.epsilon), dim=1), dim=0)

    def get_content_mul_loss(self, content_emb, content_bow):
        """
        This loss quantifies the amount of content information preserved
        in the content space
        Returns:
        cross entropy loss of the content classifier
        """
        # predictions
        preds = self.dropout(self.content_classifier(content_emb))
        # calculate cross entropy loss
        content_mul_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(preds, content_bow)

        return content_mul_loss

    def get_style_mul_loss(self, style_emb, style_labels):
        """
        This loss quantifies the amount of style information preserved
        in the style space
        Returns:
        cross entropy loss of the style classifier
        """
        # predictions
        preds = self.dropout(self.style_classifier(style_emb))
        # calculate cross entropy loss
        style_mul_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(preds, style_labels)

        return style_mul_loss

    def get_annealed_weight(self, iteration, lambda_weight):
        """
        Args:
            iteration(int): Number of iterations compeleted till now
            lambda_weight(float): KL penalty weight
        Returns:
            Annealed weight(float)
        """
        return (math.tanh(
            (iteration - mconfig.kl_anneal_iterations * 1.5) /
            (mconfig.kl_anneal_iterations / 3))
            + 1) * lambda_weight

    def get_kl_loss(self, mu, log_var):
        """
        Args:
            mu: batch of means of the gaussian distribution followed by the latent variables
            log_var: batch of log variances(log_var) of the gaussian distribution followed by the latent variables
        Returns:
            total loss(float)
        """
        kl_loss = torch.mean(-0.5*torch.sum(1+log_var -
                                             log_var.exp()-mu.pow(2), dim=1), dim=0)
        return kl_loss

    def generate_sentences(self, input_sentences, latent_emb, inference=False):
        """
        Args:
            latent_emb: generative embedding formed by the concatenation of sampled style and
                       content latent embeddings, shape = (batch_size,mconfig.generative_emb_dim)
            input_sentences: batch of token indices of input sentences, shape = (batch_size,max_seq_length)
                            It is of type 'None' when the function is called in inference mode
            inference: bool indicating whether train/inference mode
        Returns:
            output_sentences: batch of token indices or logits of generated sentences based on the
            mode of operation.
            modes:
                train: shape = (max_seq_len,batch_size,vocab_size)
                inference: shape = (max_seq_len,batch_size)
        """
        # Training mode
        if not inference:
            batch_size = input_sentences.size()[0]
            # Prepend the input sentences with <sos> token
            if input_sentences.is_cuda:
                sos_token_tensor = torch.cuda.LongTensor(
                    [gconfig.predefined_word_index['<sos>']],
                    device=input_sentences.device).unsqueeze(0).repeat(batch_size, 1)
            else:
                sos_token_tensor = torch.LongTensor(
                    [gconfig.predefined_word_index['<sos>']],
                    device=input_sentences.device).unsqueeze(0).repeat(batch_size, 1)
            input_sentences = torch.cat(
                (sos_token_tensor, input_sentences), dim=1)
            sentence_embs = self.dropout(self.decoder_embedding(input_sentences))
            # Make the latent embedding compatible for concatenation
            # by repeating it for max_seq_len + 1(additional one bcoz <sos> tokens were added)
            latent_emb = latent_emb.unsqueeze(1).repeat(
                1, mconfig.max_seq_len+1, 1)
            gen_sent_embs = torch.cat(
                (sentence_embs, latent_emb), dim=2)
            # Delete latent embedding and sos token tensor to reduce memory usage
            del latent_emb, sos_token_tensor
            output_sentences = torch.zeros(
                mconfig.max_seq_len, batch_size, mconfig.vocab_size, device=input_sentences.device)
            # initialize hidden state
            hidden_states = torch.zeros(
                batch_size, mconfig.hidden_dim, device=input_sentences.device)
            # generate sentences one word at a time in a loop
            for idx in range(mconfig.max_seq_len):
                # get words at the index idx from all the batches
                words = gen_sent_embs[:, idx, :]
                hidden_states = self.decoder(words, hidden_states)
                # project over vocab space
                next_word_logits = self.projector(hidden_states)
                output_sentences[idx] = next_word_logits
        # if inference mode is on
        else:
            if latent_emb.is_cuda:
                sos_token_tensor = torch.cuda.LongTensor(
                    [gconfig.predefined_word_index['<sos>']], device=latent_emb.device).unsqueeze(0)
            else:
                sos_token_tensor = torch.LongTensor(
                    [gconfig.predefined_word_index['<sos>']], device=latent_emb.device).unsqueeze(0)
            sentence_emb = self.decoder_embedding(sos_token_tensor)
            hidden_state = torch.zeros(
                1, mconfig.hidden_dim, device=latent_emb.device)
            # Store output sentences
            output_sentences = torch.zeros(
                mconfig.max_seq_len, 1, device=latent_emb.device)
            prev_word = sentence_emb[:,0,:]
            with torch.no_grad():
                # Greedily generate new words at a time
                for idx in range(mconfig.max_seq_len):
                    word_emb = torch.cat((prev_word, latent_emb), dim=1)
                    hidden_state = self.decoder(word_emb, hidden_state)
                    next_word_probs = nn.Softmax(dim=1)(
                        self.projector(hidden_state))
                    next_word = next_word_probs.argmax(1)
                    output_sentences[idx] = next_word
                    prev_word = self.decoder_embedding(next_word)

        return output_sentences

    def get_recon_loss(self, output_logits, input_sentences):
        """
        Args:
            output_logits: logits of output sentences at each time step, shape = (max_seq_length,batch_size,vocab_size)
            input_sentences: batch of token indices of input sentences, shape = (batch_size,max_seq_length)

        Returns:
            reconstruction loss calculated using cross entropy loss function
        """
        output_logits = output_logits.transpose(0, 1).contiguous()
        loss = nn.CrossEntropyLoss(ignore_index=0)
        recon_loss = loss(
            output_logits.view(-1, mconfig.vocab_size), input_sentences.view(-1))

        return recon_loss

    def transfer_style(self, sequence, style):
        """
        Args:
            sequence : token indices of input sentence of shape = (random_seq_length,) 
            style: target style
        Returns:
            transfered_sentence: token indices of style transfered sentence, shape=(random_seq_length,)
        """
        # pack the sequences to reduce unnecessary computations
        # It requires the sentences to be sorted in descending order to take
        # full advantage

        embedded_seq = self.encoder_embedding(sequence.unsqueeze(0))
        output, final_hidden_state = self.encoder(embedded_seq)
        sentence_emb = final_hidden_state.view(1,-1)
        # get content embeddings
        # Note that we need not calculate style embeddings since we
        # use the target style embedding
        content_emb_mu, content_emb_log_var = self.get_content_emb(
            sentence_emb)
        # sample content embeddings latent space
        sampled_content_emb = self.sample_prior(
            content_emb_mu, content_emb_log_var)
        
        # Get the approximate estimate of the target style embedding
        target_style_emb = self.avg_style_emb[style].to(device=sampled_content_emb.device)
        
        # Generative embedding
        generative_emb = self.dense(torch.cat(
            (torch.unsqueeze(target_style_emb, 0), sampled_content_emb), axis=1))
        # Generate the style transfered sentences
        transfered_sentence = self.generate_sentences(
            None, latent_emb=generative_emb, inference=True)

        return transfered_sentence.view(-1)
