
import torch
import torch.nn.functional as F


class MAML:
    def __init__(
        self,
        model,
        learning_rate=0.01,
        meta_learning_rate=0.001,
        inner_steps=2,
        device="cpu",
    ):
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.inner_steps = inner_steps
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=meta_learning_rate,
        )

    def forward(self, x, params):
        x = F.linear(
            x, params["fc1.weight"], params["fc1.bias"]
        )
        x = F.relu(x)

        x = F.linear(
            x, params["fc2.weight"], params["fc2.bias"]
        )

        return x.squeeze(-1)

    def adapt(self, x, y, params=None, first_order=False):
        if params is None:
            params = dict(self.model.named_parameters())

        adapted = dict(params)

        for _ in range(self.inner_steps):
            predictions = self.forward(x, adapted)

            loss = F.binary_cross_entropy_with_logits(
                predictions, y
            )

            gradients = torch.autograd.grad(
                loss,
                tuple(adapted.values()),
                create_graph=not first_order,
            )

            adapted = {
                name: parameter - self.learning_rate * gradient
                for (name, parameter), gradient
                in zip(adapted.items(), gradients)
            }

        return adapted

    def meta_train(self, tasks, epochs=100):
        if not tasks:
            raise ValueError("At least one training task is required.")

        self.model.train()

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss = 0.0

            for support_x, support_y, query_x, query_y in tasks:
                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                adapted_params = self.adapt(
                    support_x, support_y
                )

                predictions = self.forward(
                    query_x, adapted_params
                )

                query_loss = F.binary_cross_entropy_with_logits(
                    predictions, query_y
                )

                total_loss = total_loss + query_loss

            total_loss = total_loss / len(tasks)
            total_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"| Meta Loss: {total_loss.item():.4f}"
                )

        return self.model

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state)


def generate_user_tasks(
    number_of_users=20,
    support_size=16,
    query_size=16,
    seed=42,
):
    generator = torch.Generator().manual_seed(seed)
    tasks = []

    for _ in range(number_of_users):
        # Each simulated user has different preferences.
        user_weights = torch.randn(
            8, generator=generator
        )

        support_x = torch.rand(
            support_size, 8, generator=generator
        )

        query_x = torch.rand(
            query_size, 8, generator=generator
        )

        support_y = (
            support_x @ user_weights > 0
        ).float()

        query_y = (
            query_x @ user_weights > 0
        ).float()

        tasks.append(
            (support_x, support_y, query_x, query_y)
        )

    return tasks
