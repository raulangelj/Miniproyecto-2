"""
Miniproyecto 2 - Números Aleatorios
Raul Jimenez 19017
Donaldo Garcia 19683
"""
# %%
import math
from operator import add
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# %%
def f1(x, y):
  return x/2, y/2

def f2(x, y):
  return x/2 + 0.5, y/2

def f3(x, y):
  return x/2 + 0.25, y/2 + 0.5

# %%
# ejercicio 1.1 Cree un programa que simule 100,000 veces X para elegir entref1,f2 yf3, dibuje un triángulo de Sierpinski
# generamos el primer punto aleatorio
x = random.uniform(0, 1)
y = random.uniform(0, 1)
posible = [f1, f2, f3]
# creamos la iteracion de 100,000 veces para dibujar
for _ in range(100000):
  # elegimos una funcion aleatoria
  f = random.choice(posible)
  # calculamos el nuevo punto
  x, y = f(x, y)
  # dibujamos el punto
  plt.plot(x, y, marker='o', color='red')
# mostramos el grafico
plt.show()


# %%
# ejreicico 1.2 Determine experimentalmente p1, p2, p3 que hacen su dibujo más denso
posible = [f1, f2, f3]
# generamos el primer punto aleatorio
x = random.uniform(0, 1)
y = random.uniform(0, 1)
posible = [f1, f2, f3]
# creamos la iteracion de 100,000 veces para dibujar
for _ in range(100000):
  # elegimos una funcion aleatoria
  # f = random.choices(posible, cum_weights=(0.15,0.35, 0.50), k=1)[0]
  f = random.choices(posible, weights=(0.30,0.30, 0.40), k=1)[0]
  # calculamos el nuevo punto
  x, y = f(x, y)
  # dibujamos el punto
  plt.plot(x, y, marker='o', color='red')
# mostramos el grafico
plt.show()

# %%
# ejericcio 2.1 Cree un programa que corra el anterior juego del caos y muestre el dibujo resultante
def f1(x, y):
  return x*0.85 + y*0.04 + 0.0, x*-0.04 + y*0.85 + 1.6
def f2(x, y):
  return -0.15*x + 0.28*y + 0.0, x*0.26 + y*0.24 + 0.44
def f3(x, y):
  return x*0.2 + y*-0.26 + 0.0, x*0.23 + y*0.22 + 1.6
def f4(x, y):
  return x*0.0 + y*0.0, x*0.0 + y*0.16

F = [f1, f2, f3, f4]
P = (0.85, 0.07, 0.07, 0.01)
n = 100000
# generamos el primer punto aleatorio
x = random.uniform(0, 1)
y = random.uniform(0, 1)

# creamos la iteracion de 100,000 veces para dibujar
for _ in range(n):
  # dibujamos el punto
  plt.plot(x, y, marker='o', color='green')
  # elegimos una funcion aleatoria
  f = random.choices(F, weights=(0.85, 0.07, 0.07, 0.01), k=1)[0]
  # calculamos el nuevo punto
  x, y = f(x, y)
# mostramos el grafico
plt.show()
# %%
# ejercicio 3.1 Construya un programa que compare estos tres generadores a través de un histograma asteriscos(de 0 a 1 con saltos de 0.1) Use tres comparaciones para 100, 5,000 y 100,000 repeticiones.
def generador1(x):
  # return x = 5^5 * x mod (2 ^35  - 1)
  return 5**5 * x % (2**35 - 1)

def generador2(x):
  # return x = 7^5 * x mod (2 ^31  - 1)
  return 7**5 * x % (2**31 - 1)

def generador3(x):
  return random.random()

def create_list(n, f):
  list = []
  x = 1
  for _ in range(n):
    if f.__name__ == 'generador3':
      add_to_list = f(x)
    else:
      x = f(x)
      add_to_list = x/(2**35 - 1) if f == generador1 else x/(2**31 - 1)
    # add_to_list only will have tree decimals
    add_to_list = round(add_to_list, 3)
    list.append(add_to_list)
  return list

def generate_asterisks_frecuence_table(n, f):
  lista = create_list(n, f)
  frecuence_table = {round(i, 1): 0 for i in np.arange(0.0, 1.1, 0.1)}
  for i in np.arange(0.0, 1.1, 0.1):
    i = round(i, 1)
    for j in lista:
      if i <= j < i + 0.1:
        frecuence_table[i] += 1
  frecuence_table.pop(1)
  # if the n value is to big, we need to divede it by 10 to have a better number of asterisks
  denominador = n//100
  # print the resulta as a frecuence table with asterisks
  print( '=================== n =', n, ', f =', f.__name__, '===================')
  for item in frecuence_table:
    print(item, '-', round(item + 0.1, 1), ':', '*' * int(frecuence_table[item]/denominador), '(', frecuence_table[item], ', ', round((frecuence_table[item]*100)/n, 2) ,'%)')
  print('===============================================================')
  return frecuence_table

# %%
#  Generador 1
gen1_100 = generate_asterisks_frecuence_table(100, generador1)
gen1_5000 = generate_asterisks_frecuence_table(5000, generador1)
gen1_100000 = generate_asterisks_frecuence_table(100000, generador1)
# %%
#  Generador 2
gen2_100 = generate_asterisks_frecuence_table(100, generador2)
gen2_5000 = generate_asterisks_frecuence_table(5000, generador2)
gen2_100000 = generate_asterisks_frecuence_table(100000, generador2)
# %% 
# Generador 3
gen3_100 = generate_asterisks_frecuence_table(100, generador3)
gen3_5000 = generate_asterisks_frecuence_table(5000, generador3) 
gen3_100000 = generate_asterisks_frecuence_table(100000, generador3)

# %%
