from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any
from random import gauss
import numpy as np
import matplotlib.pyplot as plt


def test1():
    q = PriorityQueue()

    for _ in range(500):
        q.put(gauss(0, 1))

    ns = []

    for _ in range(5000):
        min_item = q.get()

        second_min_item = q.get()
        q.put(second_min_item)

        # x = gauss(0, 1)
        # q.put(x)
        # while x >= second_min_item:
        #     x = gauss(0, 1)
        #     q.put(x)

        n = 1
        x = gauss(0, 1)
        while x >= second_min_item:
            x = gauss(0, 1)
            n += 1
        q.put(x)

        ns.append(n)

    print(np.mean(ns), np.std(ns))

    items = []
    while not q.empty():
        items.append(q.get())


    print(np.mean(items), np.std(items))

    plt.hist(items, bins='auto')

    plt.figure()
    plt.hist(ns, bins='auto')

    plt.show()



@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

class Painter:
    @dataclass(order=True)
    class It:
        priority: int
        item: Any=field(compare=False)

def test2():
    q = PriorityQueue()
    item1 = PrioritizedItem(priority=10.0, item='help')
    item2 = PrioritizedItem(priority=5.0, item='zadd')
    q.put(item1)
    q.put(item2)

    x = q.get()

    print(x)


if __name__ == '__main__':
    test2()

    print(Painter.It)
    x = Painter()
    print(x.It)


    d = {
        'a': 4,
        'b': {
            'y': 8
        }
    }

    dc = d.copy()

    d['b']['y'] = -100

    print(d)
    print(dc)




