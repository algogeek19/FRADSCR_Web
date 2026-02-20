def mulatu(n):
    if n == 0: return 4
    elif n == 1: return 1
    return mulatu(n - 1) + mulatu(n - 2)
def fib(n):
    if n <= 1:return 1
    return fib(n - 1) + fib(n - 2)
import matplotlib.pyplot as plt
dictionary = {}
dit = {}
for n in range(0, 11):
    if n not in dictionary:
        dictionary[n] = mulatu(n)
for i in range(0, 151):
    if i not in dit:
        dit[i]  = fib(i)
print(dictionary, dit)

key11 = [keys for keys in dictionary]
value11 = [values for values in dictionary.items()]
plt.plot(key11, value11)
value = [values for values in dit.items()]
key = [i for i in dit]
plt.plot(key, value)

plt.show()