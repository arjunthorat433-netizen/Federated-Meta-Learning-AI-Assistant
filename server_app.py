
import torch

from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from federated_app.task import ClientModel, test


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):

    print("FEDERATED SERVER STARTED")

    
    num_rounds = context.run_config["num-server-rounds"]

   
    global_model = ClientModel()
    initial_arrays = ArrayRecord(global_model.state_dict())

  
    strategy = FedAvg(fraction_evaluate=1.0)

    print("FEDAVG STRATEGY INITIALIZED")

    
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    
    final_state = result.arrays.to_torch_state_dict()

    torch.save(final_state, "final_model.pt")

    print("GLOBAL MODEL SAVED")
    print("FEDERATED TRAINING COMPLETED")


def global_evaluate(server_round: int, arrays: ArrayRecord):

    model = ClientModel()
    model.load_state_dict(arrays.to_torch_state_dict())

    
    accuracies = []

    for client_id in range(1, 4):
        accuracy = test(model, client_id)
        accuracies.append(accuracy)

    avg_accuracy = sum(accuracies) / len(accuracies)

    print(
        f"Round {server_round}, "
        f"Global Accuracy: {avg_accuracy:.4f}"
    )

    return MetricRecord({
        "accuracy": avg_accuracy
    })
