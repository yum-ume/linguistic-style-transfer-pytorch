import torch
from torch.utils.data import DataLoader
from linguistic_style_transfer_pytorch.config import GeneralConfig, ModelConfig
from linguistic_style_transfer_pytorch.data_loader import TextDataset
from linguistic_style_transfer_pytorch.model import AdversarialVAE
from tqdm import tqdm, trange
import os
import numpy as np
import pickle

use_cuda = True if torch.cuda.is_available() else False
if use_cuda:
    device='cuda:7'
seed = 10
np.random.seed(seed)
torch.manual_seed(seed)


if __name__ == "__main__":

    mconfig = ModelConfig()
    gconfig = GeneralConfig()    
    weights = torch.FloatTensor(np.load(gconfig.word_embedding_path))
    model = AdversarialVAE(weight=weights)
    if use_cuda:
        model = model.cuda(device=device)
        torch.cuda.manual_seed(seed)
        
    #=============== Define dataloader ================#
    train_dataset = TextDataset(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=mconfig.batch_size, shuffle=True)
    content_discriminator_params, style_discriminator_params, vae_and_classifier_params = model.get_params()
    #============== Define optimizers ================#
    # content discriminator/adversary optimizer
    content_disc_opt = torch.optim.RMSprop(
        content_discriminator_params, lr=mconfig.content_adversary_lr)
    # style discriminaot/adversary optimizer
    style_disc_opt = torch.optim.RMSprop(
        style_discriminator_params, lr=mconfig.style_adversary_lr)
    # autoencoder and classifiers optimizer
    vae_and_cls_opt = torch.optim.Adam(
        vae_and_classifier_params, lr=mconfig.autoencoder_lr)
    print("Training started!")
    for epoch in trange(mconfig.epochs, desc="Epoch"):

        for iteration, batch in enumerate(tqdm(train_dataloader)):

            # unpacking
            sequences, seq_lens, labels, bow_rep = batch
            if use_cuda:
                sequences = sequences.cuda(device=device)
                seq_lens = seq_lens.cuda(device=device)
                labels = labels.to(torch.float).cuda(device=device)
                bow_rep = bow_rep.cuda(device=device)
            content_disc_loss, style_disc_loss, _ = model(
                sequences, seq_lens.squeeze(1), labels, bow_rep, iteration+1, epoch == mconfig.epochs-2)

            #============== Update Adversary/Discriminator parameters ===========#
            # update content discriminator parametes
            # we need to retain the computation graph so that discriminator predictions are
            # not freed as we need them to calculate entropy.
            # Note that even even detaching the discriminator branch won't help us since it
            # will be freed and delete all the intermediary values(predictions, in our case).
            # Hence, with no access to this branch we can't backprop the entropy loss
            model.zero_grad()
            content_disc_loss.backward()
            content_disc_opt.step()

            # update style discriminator parameters
            style_disc_loss.backward()
            style_disc_opt.step()

            #=============== Update VAE and classifier parameters ===============#
            _, _, vae_and_cls_loss = model(
                sequences, seq_lens.squeeze(1), labels, bow_rep, iteration+1, epoch == mconfig.epochs-1)
            
            model.zero_grad()
            vae_and_cls_loss.backward()
            vae_and_cls_opt.step()
            
        print("Saving states")
        #================ Saving states ==========================#
        if not os.path.exists(gconfig.model_save_path):
            os.mkdir(gconfig.model_save_path)
        # save model state
        torch.save(model.state_dict(),
                   gconfig.model_save_path + f'/model_epoch_{epoch+1}.pt')
        # save optimizers states
        torch.save({'content_disc': content_disc_opt.state_dict(),
                    'style_disc': style_disc_opt.state_dict(),
                    'vae_and_cls': vae_and_cls_opt.state_dict()},
                   gconfig.model_save_path + f'/opt_epoch_{epoch+1}.pt')
    # Save approximate estimate of different style embeddings after the last epoch
    with open(gconfig.avg_style_emb_path, 'wb') as f:
        style_embedding_map = dict()
        for label, each_style_embs in model.style_embedding_map.items():
            style_embedding_map[label] = torch.cat(model.style_embedding_map[label], dim=0)
        avg_style_emb = dict()
        for label,each_style_embs_tensor in style_embedding_map.items():
            avg_style_emb[label] = torch.mean(each_style_embs_tensor, dim=0)
        pickle.dump(avg_style_emb, f)
    print("Training completed!!!")
