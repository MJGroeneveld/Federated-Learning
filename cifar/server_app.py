from typing import List, Tuple
import logging
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from cifar.functions import get_parameters
from cifar.task import ClassifierCIFAR10

# logging.basicConfig(level=logging.INFO)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:

    examples = [num_examples for num_examples, _ in metrics]

    # Accuracy
    accuracies = [
        num_examples * m.get("accuracy", 0.0) 
        for num_examples, m in metrics
    ]
    avg_accuracy = sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0

    # AUROC (bijvoorbeeld gewogen gemiddelde)
    aurocs = [
        num_examples * m.get("auroc_ood", 0.0)
        for num_examples, m in metrics
    ]
    avg_auroc = sum(aurocs) / sum(examples) if sum(examples) > 0 else 0.0

    return {
        "accuracy": avg_accuracy,
        "auroc_ood": avg_auroc
    }
    # # Multiply accuracy of each client by number of examples used
    # accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    # examples = [num_examples for num_examples, _ in metrics]

    # # Aggregate and return custom metric (weighted average)
    # return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""

    # Convert model parameters to flwr.common.Parameters
    ndarrays = get_parameters(ClassifierCIFAR10())
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        initial_parameters=global_model_init,
        evaluate_metrics_aggregation_fn=weighted_average, 
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config

    )
    # Construct ServerConfig
    num_rounds = 10 #context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds, round_timeout=None)

    return ServerAppComponents(strategy=strategy, config=config)

def fit_config(server_round: int):
    """Generate training configuration for each round."""
    # Create the configuration dictionary
    config = {
        "batch_size"        : 32,
        "current_round"     : server_round,
        "local_epochs"      : 50,
        "num_classes"       : 10, 
        'bin'               : 'ood_federated_learning', 
        'experiment_name'   : '23092025',
    }
    return config

def evaluate_config(server_round: int):
    """Generate evaluation configuration for each round."""
    # Create the configuration dictionary
    config = {
        "batch_size"        : 32,
        "current_round"     : server_round,
        "local_epochs"      : 50,
        "num_classes"       : 10, 
        # "metrics": ["accuracy"],  # Example metrics to compute
        'bin'               : 'ood_federated_learning', 
        'experiment_name'   : '23092025',
    }
    return config

app = ServerApp(server_fn=server_fn)