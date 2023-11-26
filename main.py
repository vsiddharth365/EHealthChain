from ACO_Data_Offloading import ACO_DataOffloading
from ACO_Blockchain_Optimization import ACO_BlockchainOptimization
from variableInitialization import initialize
from blockchain import startBlockchain
import threading

lock = threading.Lock()  # shared lock


def ACO_algorithm1():
    with lock:
        initialize()  # initializing all variables defined in problem statement
        ACO_DataOffloading()  # run the proposed ACO algorithm to minimize data offloading cost


def ACO_algorithm2():
    with lock:
        ACO_BlockchainOptimization()  # run the proposed ACO algorithm to minimize blockchain utility
        startBlockchain()  # once the EHRs have been processed, store them into blockchain


if __name__ == "__main__":
    # Create threads for both algorithms
    thread1 = threading.Thread(target=ACO_algorithm1)
    thread2 = threading.Thread(target=ACO_algorithm2)
    # Start the threads
    thread1.start()
    thread2.start()
    # Wait for both threads to finish
    thread1.join()
    thread2.join()
