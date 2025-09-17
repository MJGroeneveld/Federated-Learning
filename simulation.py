# server.py

from cifar.client_app import client   # importeer je ClientApp
from cifar.server_app import app    # importeer je serverapp 
from flwr.simulation import run_simulation

if __name__ == "__main__":
    run_simulation(
        client_app=client,
        server_app=app,
        num_supernodes=3,   # aantal clients die je wilt simuleren
    )
