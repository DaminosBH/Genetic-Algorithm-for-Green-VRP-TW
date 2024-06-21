# -*- coding: utf-8 -*-


import random
from gavrptw.core import run_gavrptw


def main():
    '''main()'''
    random.seed(35121711)

    instance_name = 'C101'
    unit_cost = 8.0
    init_cost = 100.0
    wait_cost = 1.0
    delay_cost = 1.5
    carbon_cost=0.056

    ind_size = 100
    pop_size =30
    cx_pb = 0.9
    mut_pb = 0.1
    n_gen = 2000
    a,b,c,d,e,f,g,v=4529.765238, 60.24714667, 0.289025869, 0.016975237, -0.000108584, 9.74901e-07, 0, 50.0
    export_csv = True
  
    run_gavrptw(carbon_cost=carbon_cost,a=a,b=b,c=c,d=d,e=e,f=f,g=g,v=v,instance_name=instance_name, unit_cost=unit_cost, init_cost=init_cost,  \
        wait_cost=wait_cost, delay_cost=delay_cost, ind_size=ind_size, pop_size=pop_size, \
        cx_pb=cx_pb, mut_pb=mut_pb, n_gen=n_gen, export_csv=export_csv)


if __name__ == '__main__':
    main()
