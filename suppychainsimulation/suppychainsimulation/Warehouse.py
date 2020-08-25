import simpy
import numpy as np 
from products import Product
# Stocking facility class
class stockingFacility(object):
    # initialize the new facility object
    def __init__(self, 
                env, 
                productDict,
                initialInv, 
                ROP, 
                ROQ, 
                warehouseCapacity,
                warningCapPoint = 0.9,
                stdLeadTime = 0.5):
        self.env = env

        self.productDict = productDict
        self.productName = list(self.productDict.keys())
        self.warehouseCapacity = warehouseCapacity
        self.warningCapacity = self.warehouseCapacity * warningCapPoint
        self.stdLeadTime = stdLeadTime 

        self.on_hand_inventory = initialInv.copy()  
        self.inventory_position = initialInv.copy()
        self.ROP = ROP.copy()  # reorder point
        self.ROQ = ROQ.copy()   # reorder quantity (lot size)

        # initialize the metrics
        self.onhandVolume = 0
        self.totalDemand = dict()
        self.totalBackOrder = dict()
        self.totalLateSales = dict()
        self.serviceLevel = dict()
        
        self.TotalCost = dict()
        self.curOrderOnhand = dict()
        self.curOrderPosition = dict()
        
        for productname, product in self.productDict.items():
            self.onhandVolume += product.volume * initialInv[productname]
            self.totalDemand[productname] = 0.0
            self.totalBackOrder[productname] = 0.0
            self.totalLateSales[productname] = 0.0
            self.serviceLevel[productname] = 0.0
            
            self.TotalCost[productname] = product.price * self.on_hand_inventory[productname]
            self.curOrderOnhand[productname] = []
            self.curOrderPosition[productname] = []
        
        # start the processes
        self.env.process(self.check_inventory())
        self.env.process(self.serve_customer())

    # main subroutine for facility operation
    # it records all stocking metrics for the facility
    def serve_customer(self):
        while True:
            #print("serve", self.env.now)
            yield self.env.timeout(1.0)
            for product_name, product in self.productDict.items():
                demand = product.generate_demand()
                # print(demand,product_name)
                self.totalDemand[product_name] += demand
                shipment = min(demand + self.totalBackOrder[product_name],self.on_hand_inventory[product_name])
                self.on_hand_inventory[product_name] -= shipment
                self.inventory_position[product_name] -= demand
                backorder = demand - shipment
                self.totalBackOrder[product_name] += backorder
                self.totalLateSales[product_name] += max(0.0, backorder)
                #print("inventory_position,", self.inventory_position)
                #print("on_hand_inventory,", self.on_hand_inventory)
                self.curOrderOnhand[product_name].append(self.on_hand_inventory[product_name])
                self.curOrderPosition[product_name].append(self.inventory_position[product_name])
                self.onhandVolume -= shipment * product.volume



    
    
    # process to place replenishment order
    def check_inventory(self):
        while True:
            #print("check", self.env.now)
            yield self.env.timeout(7.0)
            for product_name, product in self.productDict.items():
                if self.inventory_position[product_name] <= 1.01 * self.ROP[product_name]: 
                    self.env.process(self.ship(product, self.ROQ[product_name]))
                    self.inventory_position[product_name] += self.ROQ[product_name]
                    self.TotalCost[product_name] += product.price * self.ROQ[product_name]
                #print("inventory_position,", self.inventory_position)
                #print("on_hand_inventory,", self.on_hand_inventory)


    # subroutine for a new order placed by the facility
    def ship(self, product, orderQty):
        leadTime = int(np.random.normal(product.leadTime, self.stdLeadTime, 1))
        # print(product.get_name(), leadTime)
        # print("lead time", leadTime)
        yield self.env.timeout(leadTime)  # wait for the lead time before delivering
        temp = self.onhandVolume + product.volume * orderQty
        if self.warehouseCapacity >temp >= self.warningCapacity:
            # print(self.env.now, "warning!", product.get_name())
            # only keep half of demands
            self.on_hand_inventory[product.get_name()] += int(orderQty/2)
            self.onhandVolume += product.volume * int(orderQty/2)
        elif temp < self.warningCapacity:
            self.on_hand_inventory[product.get_name()] += int(orderQty)
            self.onhandVolume += product.volume * int(orderQty)    
        #else:
            # print(self.env.now,"dead!", product.get_name())     