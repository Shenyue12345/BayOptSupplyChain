import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import random

class Product():
    def __init__(self, data, product_name, price, leadtime, volume):
        self.data = data
        self.product_name = product_name
        self.price = price
        self.leadTime = leadtime
        self.volume = volume
        self.iterator = data.iterrows()
    
    def get_name(self):
        return self.product_name
    
    def generate_demand(self):
        return next(self.iterator)[1][0]

def generate_products(ndays, plotdata = False):
    # random.seed(0)
    df1 = pd.DataFrame({"demand":np.random.gamma(0.4, 3, size=ndays).astype(int)})
    product1 = Product(data = df1, product_name = "Product1", price = 3200, leadtime = 17, volume = 18)

    df2 = pd.DataFrame({"demand":np.random.gamma(2, 3, size=ndays).astype(int)})
    product2 = Product(data = df2, product_name = "Product2", price = 2400, leadtime = 30, volume = 25)
    
    df3 = pd.DataFrame({"demand":np.random.normal(7, 0.5, size=ndays).astype(int)})
    product3 = Product(data = df3, product_name = "Product3", price = 1200, leadtime = 6, volume = 7)
    
    df4 = pd.DataFrame({"demand":np.random.normal(23, 4, size=ndays).astype(int)})
    product4 = Product(data = df4, product_name = "Product4", price = 500, leadtime = 8, volume = 63)
    
    df5 = pd.DataFrame({"demand":np.random.normal(6, 3, size=ndays).astype(int)})
    product5 = Product(data = df5, product_name = "Product5", price = 73, leadtime = 14, volume = 3)
    
    df6 = pd.DataFrame({"demand":np.random.normal(47, 12, size=ndays).astype(int)})
    product6 = Product(data = df6, product_name = "Product6", price = 14, leadtime = 3, volume = 12)

    if plotdata == True:
        plt.figure(figsize = (16,8))
        df1["demand"].hist(alpha = 0.5, label = "Product1")
        df2["demand"].hist(alpha = 0.5, label = "Product2")
        df3["demand"].hist(alpha = 0.5, label = "Product3")
        df4["demand"].hist(alpha = 0.5, label = "Product4")
        df5["demand"].hist(alpha = 0.5, label = "Product5")
        df6["demand"].hist(alpha = 0.5, label = "Product6")
        plt.legend()
    return {"Product1": product1,
            "Product2": product2,
            "Product3": product3,
            "Product4": product4,
            "Product5": product5,
            "Product6": product6}
    