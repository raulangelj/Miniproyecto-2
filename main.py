"""
Miniproyecto 2 - Números Aleatorios
Raul Jimenez 19017
Donaldo Garcia 19683
"""
# %%
import matplotlib.pyplot as plt
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