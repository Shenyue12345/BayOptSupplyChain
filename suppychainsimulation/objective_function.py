from products import Product, generate_products
import simpy
from Warehouse import stockingFacility
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult

def objective_function_penalty(x, penalty, target_service_level,**kwargs):
    WarehouseA = get_sim_result(x, **kwargs)
    totalcost = 0
    penaltycost = 0
    for i in range(1,7):
        totalcost += WarehouseA.TotalCost["Product" + str(i)]
        servicelevel = 1 - WarehouseA.totalLateSales["Product" + str(i)]/ WarehouseA.totalDemand["Product" + str(i)]
        if target_service_level > servicelevel:
            penaltycost += (target_service_level - servicelevel) * WarehouseA.totalDemand["Product" + str(i)] * penalty
            
    return totalcost + penaltycost

def objective_function_KKT(x, target_service_level,**kwargs):
    WarehouseA = get_sim_result(x,  **kwargs)
    
    totalcost = 0
    penaltycost = 0
    for i in range(1,7):
        totalcost += WarehouseA.TotalCost["Product" + str(i)]
        servicelevel = 1 - WarehouseA.totalLateSales["Product" + str(i)]/ WarehouseA.totalDemand["Product" + str(i)]
        penaltycost += (target_service_level - servicelevel) *  x[11 + i]
            
    return totalcost + penaltycost

def objective_function(x, return_SL = False, **kwargs):
    WarehouseA = get_sim_result(x,**kwargs)
    totalcost = 0
    if return_SL == False:
        for i in range(1,7):
            totalcost += WarehouseA.TotalCost["Product" + str(i)]
    if return_SL == True:
        min_SL = 1
        for i in range(1,7):
            servicelevel = 1 - WarehouseA.totalLateSales["Product" + str(i)]/ WarehouseA.totalDemand["Product" + str(i)]
            min_SL = min(min_SL, servicelevel)
    return totalcost if return_SL == False else min_SL

def objective_function_products(x, return_SL, **kwargs):
    WarehouseA = get_sim_result(x,**kwargs)
    if return_SL:
        sL = []
        for i in range(1,7):
            servicelevel = 1 - WarehouseA.totalLateSales["Product" + str(i)]/ WarehouseA.totalDemand["Product" + str(i)]
            sL.append(servicelevel)
            return sL 
    else: 
        min_SL = 1e11
        for i in range(1,7):
            servicelevel = WarehouseA.totalDemand["Product" + str(i)] * 0.2 - WarehouseA.totalLateSales["Product" + str(i)]
            min_SL = min(servicelevel, min_SL)
        
        return min_SL



def get_sim_result(x, days, plotdata):
    productDict = generate_products(days, plotdata)
    ROP = {"Product1": x[0],
        "Product2": x[1],
        "Product3": x[2],
        "Product4": x[3],
        "Product5": x[4],
        "Product6": x[5]}
    ROQ = {"Product1": x[6],
        "Product2": x[7],
        "Product3": x[8],
        "Product4": x[9],
        "Product5": x[10],
        "Product6": x[11]}
    initialInv = dict()
    for product_name in ROP.keys():
        initialInv[product_name] = ROP[product_name] + ROQ[product_name]


    env = simpy.Environment()
    WarehouseA = stockingFacility(env, 
                                productDict = productDict,
                                initialInv = initialInv, 
                                    ROP = ROP, 
                                    ROQ = ROQ, 
                                    warehouseCapacity = 100000,
                                    warningCapPoint = 0.8,
                                    stdLeadTime = 0.5)
    env.run(until=days)
    return WarehouseA











def plot_warehouse(warehouse,days):
    plt.figure(figsize = (16,16))
    sumcost = 0
    for i in range(1,7):
        plt.subplot(3,2,i)
        plt.plot(list(range(days - 1)), warehouse.curOrderOnhand["Product" + str(i)], label = "current products on hand")
        plt.plot(list(range(days - 1)), warehouse.curOrderPosition["Product" + str(i)], "-.",label = "ideal products on hand")
        plt.plot(list(range(days - 1)), [warehouse.ROP["Product" + str(i)]] * (days - 1), label = "reorder point")
        plt.legend()
        servicelevel = 1 - warehouse.totalLateSales["Product" + str(i)]/ warehouse.totalDemand["Product" + str(i)]
        totalcost = warehouse.TotalCost["Product" + str(i)]
        plt.title("Product" + str(i) + "service level is " + str(servicelevel) + ". Total cost is " + str(totalcost))
        sumcost += totalcost
    print("all cost is: ", sumcost)


def print_result(days, r1):
    
    if isinstance(r1, OptimizeResult):
        print("the minimize cost is,", r1.fun)
        print("the best parameter is",r1.x)
        plot_convergence(r1)
        WarehouseA = get_sim_result(r1.x, days, plotdata = True)
        plot_warehouse(WarehouseA,days)
    else:
        plt.plot(list(range(r1.niter)), [np.min(r1.y_obj[0:x]) for x in range(1,1+  r1.niter)])
        idx = np.where(r1.y_constraint > 0)[0]
        plt.plot(idx - r1.n_restarts + 1, r1.y_obj[idx], "r.", label = "service level > 0.8")
        plt.xlabel('number of calls n')
        plt.ylabel('min f(x) after n calls', fontsize=16)
        plt.title("Convergence plot")
        plt.legend()

        WarehouseA = get_sim_result(r1.min_x, days, plotdata = True)
        plot_warehouse(WarehouseA,days)

