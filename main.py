from ACO_Data_Offloading import ACO_DataOffloading
from ACO_Blockchain_Optimization import ACO_BlockchainOptimization
from variableInitialization import initialize
from blockchain import startBlockchain

if __name__ == "__main__":
    initialize()  # initializing all variables defined in problem statement
    ACO_DataOffloading()  # run the proposed ACO algorithm to solve the objective function
    ACO_BlockchainOptimization()
    startBlockchain()  # once the EHRs have been processed, store them into blockchain
