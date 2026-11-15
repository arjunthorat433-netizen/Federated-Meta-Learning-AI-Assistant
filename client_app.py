
import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated_app.task import ClientModel, train, test


app = ClientApp()


@app.train()
def train_client(msg: Message, context: Context):

   
    model = ClientModel()
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

  
    client_id = int(context.node_config["partition-id"]) + 1

   
    train_loss = train(model, client_id, epochs=5)

    
    model_record = ArrayRecord(model.state_dict())

    metrics = MetricRecord({
        "train_loss": train_loss,
        "num-examples": 100,
    })

    content = RecordDict({
        "arrays": model_record,
        "metrics": metrics,
    })

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_client(msg: Message, context: Context):

    
    model = ClientModel()
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    client_id = int(context.node_config["partition-id"]) + 1

   
    accuracy = test(model, client_id)

    metrics = MetricRecord({
        "eval_acc": accuracy,
        "num-examples": 100,
    })

    content = RecordDict({
        "metrics": metrics,
    })

    return Message(content=content, reply_to=msg)
