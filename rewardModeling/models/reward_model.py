import time

class RewardModelTrainer:
    def __init__(self, model, device, optimizer, loss_fn):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, dataloader, epochs=3):
        self.model.train()
        total_start = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start = time.time()

            for step, batch in enumerate(dataloader):
                step_start = time.time()

                # Move data to device
                preferred_input_ids = batch["preferred_input_ids"].to(self.device)
                preferred_attention_mask = batch["preferred_attention_mask"].to(self.device)
                non_preferred_input_ids = batch["non_preferred_input_ids"].to(self.device)
                non_preferred_attention_mask = batch["non_preferred_attention_mask"].to(self.device)
                labels = batch["label"].to(self.device).float()

                # Forward pass
                preferred_rewards = self.model(input_ids=preferred_input_ids, attention_mask=preferred_attention_mask).logits
                non_preferred_rewards = self.model(input_ids=non_preferred_input_ids, attention_mask=non_preferred_attention_mask).logits
                logits = (preferred_rewards - non_preferred_rewards).squeeze(-1)
                loss = self.loss_fn(logits, labels)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                step_time = time.time() - step_start

                if (step + 1) % 10 == 0 or (step + 1) == len(dataloader):
                    print(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(dataloader)}] "
                          f"Loss: {loss.item():.4f} - Step Time: {step_time:.2f}s")

            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}] done in {epoch_time:.2f}s - Avg Loss: {avg_loss:.4f}\n")

        total_time = time.time() - total_start
        print(f"Total training time for {epochs} epoch(s): {total_time:.2f} seconds.")
