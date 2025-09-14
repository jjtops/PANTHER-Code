from utils import*
from ds import*
from tqdm import tqdm





def train():
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        iter_queue = iter(queue)

        loop = tqdm(range(len(queue)), leave=False, desc=f"Epoch[{epoch+1}/{epochs}")

        for _ in loop:
            batch = next(iter_queue)
            img = batch['image'][tio.DATA].float()  # (C, H, W, D) (0, 1, 2, 3)
            mask = batch['label'][tio.DATA].float()


            img = img.to(device)
            mask = mask.to(device)

            img = img.permute(0, 3, 1, 2) # (C, D, H, W)
            mask = mask.permute(0, 3, 1, 2)

            img = img.unsqueeze(0) # (B, C, D, H, W)
            mask = mask.unsqueeze(0)


            optimizer.zero_grad()
            logits = model(img)

            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch[{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(queue)
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_epoch_loss:.4f}")
        val2()

valNum = 1
def val2():

    with torch.no_grad():
        val_loss = 0.0
        metric.reset()

        for subj in subjectsVal:
            grid_sampler = tio.inference.GridSampler(
                subj,
                patch_size=(96, 96, 48),
                patch_overlap=(24, 24, 12)
            )
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
            aggregator = tio.inference.GridAggregator(grid_sampler)

            for patches_batch in patch_loader:
                img = patches_batch['image'][tio.DATA].float().to(device)
                locations = patches_batch[tio.LOCATION]
                img = img.permute(0, 1, 4, 2, 3) #[4, 1, 96, 96, 48] (0, 1, 2, 3, 4) --> [4, 1, 48, 96, 96] == (0, 1, 4, 2, 3)

                logits = model(img)
                logits = logits.permute(0, 1, 3, 4, 2)
                aggregator.add_batch(logits, locations)


            # reconstruct full volume prediction
            full_logits = aggregator.get_output_tensor().unsqueeze(0).to(device)
            mask = subj['label'][tio.DATA].unsqueeze(0).to(device)
            # print("mask: ", mask.shape, "logits: ", full_logits.shape)

            mask = mask.permute(0, 1, 4, 2, 3)
            full_logits = full_logits.permute(0, 1, 4, 2, 3)

            loss = criterion(full_logits, mask)
            val_loss += loss.item()

            pred = torch.sigmoid(full_logits) > 0.5
            metric(y_pred=pred.float(), y=mask)

        avg_val_loss = val_loss / len(subjectsVal)
        mean_dice = metric.aggregate().item()

        scheduler.step(avg_val_loss)

        print(f"Val Number{valNum} - Average Loss: {avg_val_loss:.4f} Mean Dice: {mean_dice:.4f}")


            # ---- SAVE MODEL IF BETTER ----
        if mean_dice > utils.best_score:
            utils.best_score = mean_dice
            benchmark = {
                'model_state_dict': model.state_dict(),
                'loss': mean_dice,
                'optimizer_state_dict': optimizer.state_dict(),
            }

            torch.save(benchmark, "bestPre2.pth")
            print(f"✅ Saved new best model with loss {utils.best_score:.4f}")




if __name__ == "__main__":
    # val2()
    train()










