
import torch
import torch.nn as nn
import torch.optim as optim

from federated_app.task import ClientModel, load_data


def maml_train(
    model,
    client_ids=(1, 2, 3),
    meta_epochs=20,
    inner_lr=0.01,
    meta_lr=0.001,
    inner_steps=5,
):
    meta_optimizer = optim.Adam(
        model.parameters(),
        lr=meta_lr
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(meta_epochs):

        meta_optimizer.zero_grad()

        for client_id in client_ids:

            X, y = load_data(client_id)

            
            generator = torch.Generator().manual_seed(client_id)
            indices = torch.randperm(
                len(X),
                generator=generator
            )

            split = int(0.8 * len(X))

            support_idx = indices[:split]
            query_idx = indices[split:]

            X_support = X[support_idx]
            y_support = y[support_idx]

            X_query = X[query_idx]
            y_query = y[query_idx]

           
            adapted_model = ClientModel()
            adapted_model.load_state_dict(model.state_dict())

            inner_optimizer = optim.SGD(
                adapted_model.parameters(),
                lr=inner_lr
            )

           
            adapted_model.train()

            for _ in range(inner_steps):

                inner_optimizer.zero_grad()

                outputs = adapted_model(X_support)
                support_loss = criterion(
                    outputs,
                    y_support
                )

                support_loss.backward()
                inner_optimizer.step()

            
            adapted_model.zero_grad(set_to_none=True)
            adapted_model.eval()

            query_outputs = adapted_model(X_query)

            query_loss = criterion(
                query_outputs,
                y_query
            )

            query_loss.backward()

            
            meta_params = dict(model.named_parameters())
            adapted_params = dict(adapted_model.named_parameters())

            for name, meta_param in meta_params.items():

                adapted_grad = adapted_params[name].grad

                if adapted_grad is not None:

                    if meta_param.grad is None:
                        meta_param.grad = adapted_grad.detach().clone()
                    else:
                        meta_param.grad.add_(
                            adapted_grad.detach()
                        )

       
        for param in model.parameters():

            if param.grad is not None:
                param.grad.div_(len(client_ids))

       
        meta_optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(
                f"Meta Epoch {epoch + 1}/{meta_epochs} completed"
            )

    return model


if __name__ == "__main__":

    model = ClientModel()

    
    state_dict = torch.load(
        "final_model.pt",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    print("FEDERATED MODEL LOADED!")

    trained_model = maml_train(model)

    torch.save(
        trained_model.state_dict(),
        "maml_model.pt"
    )

    print("MAML MODEL SAVED SUCCESSFULLY!")
