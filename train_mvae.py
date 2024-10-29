from models.mvae import loss_function
import wandb

wandb.require("core")


def train_network(model, data_loader, optimizer, num_epochs, num_components):
    model.train()
    loss_train_history = []
    use_wandb = True
    if use_wandb:
        wandb.login()
        run = wandb.init(project="MVAE")

        wandb.run.log_code(
            ("/home/sheida.rahnamai/GIT/HDN/"),
            include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
        )
    
    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, _ in data_loader:
            optimizer.zero_grad()

            # Forward pass
            x_reconstructed, mu, logvar, component_idx = model(x_batch)

            # Compute the loss
            loss = loss_function(
                x_batch, x_reconstructed, mu, logvar, component_idx, num_components
            )
            total_loss += loss.item()
            if use_wandb:
                run.log(
                    {
                        "global_idx": global_idx,
                        "idx": idx,
                        "IP": inpainting_loss * alpha,
                        "KL": kl_loss * beta,
                        "CL": cl_loss * gamma,
                        "PPL": cl_pos,
                        "NPL": cl_neg,
                        "Total": loss,
                    },
                    commit=True,
                )

            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader.dataset)}")
