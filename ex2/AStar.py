import bisect
import heapq
from abc import ABC, abstractmethod

import numpy as np
class Queue(ABC):
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(lt): Queue where items are sorted by lt, (default <).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def update(x, **entries):
        """Update a dict; or an object with slots; according to entries.
        # >>> update({'a': 1}, a=10, b=20)
        # {'a': 10, 'b': 20}
        # >>> update(Struct(a=1), a=10, b=20)
        Struct(a=10, b=20)
        """
        if isinstance(x, dict):
            x.update(entries)
        else:
            x.__dict__.update(entries)
        return x

    @abstractmethod
    def __init__(self):
        pass


    def extend(self, items):
        for item in items: self.append(item)

class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x)."""

    def __init__(self, order=min, f=lambda x: x):
        self.update( A=[], order=order, f=f)

    def append(self, item):

        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.update(self, state=state, parent=parent, action=action,
               path_cost=path_cost, depth=0)
        if parent:
            self.depth = parent.depth + 1
        self.state = state
        self.action = action
        self.path_cost = path_cost


    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1


    def update(self,x, **entries):
        """Update a dict; or an object with slots; according to entries.
        update({'a': 1}, a=10, b=20)
        {'a': 10, 'b': 20}
         update(Struct(a=1), a=10, b=20)
        Struct(a=10, b=20)
        """
        if isinstance(x, dict):
            x.update(entries)
        else:
            x.__dict__.update(entries)
        return x
    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def path(self):
        "Create a list of nodes from the root to this node."
        x, result = self, [self]
        while x.parent:
            result.append(x.parent)
            x = x.parent
        return result

    def expand(self, problem):
        "Return a list of nodes reachable from this node. [Fig. 3.8]"
        l =  [Node(next, self, act,
                     problem.path_cost(self.path_cost, self.state, act, next))
                for (act, next) in problem.successor(self.state)]
        return  l

    def __eq__(self, other):
        return (self.f == other.f)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return (self.f < other.f)

    def __gt__(self, other):
        return (self.f > other.f)

    def __le__(self, other):
        return (self < other) or (self == other)

    def __ge__(self, other):
        return (self > other) or (self == other)


class AStar:
    def __init__(self,problem,f):
        self.fringe = PriorityQueue(min, f)

    def graph_search(self,problem, fringe,initial = None):
        """Search through the successors of a problem to find a goal.
        The argument fringe should be an empty queue.
        If two paths reach a state, only use the best one. [Fig. 3.18]"""
        closed = {}
        expanded = 0
        if initial:
            fringe.append((Node(initial)))
        else:
            fringe.append(Node(problem.initial))
        while fringe:
            node = fringe.pop()
            if problem.goal_test(node.state):
                return node, expanded
            if node.state not in closed:
                closed[node.state] = True
                fringe.extend(node.expand(problem))
                expanded += 1
        return None

    def search(self,problem, h=None,initial_state=None):
        """A* search is best-first graph search with f(n) = g(n)+h(n).
        You need to specify theh function when you call astar_search.
        Uses the pathmax trick: f(n) = max(f(n), g(n)+h(n))."""
        h = h or problem.h

        def f(n):

            return max(getattr(n, 'f', -np.infty), n.path_cost + h(n))

        f = self.memoize(f, 'f')
        return self.graph_search(problem, PriorityQueue(min, f),initial_state)

    def memoize(self,fn, slot=None):
        """Memoize fn: make it remember the computed value for any argument list.
        If slot is specified, store result in that slot of first argument.
        If slot is false, store results in a dictionary."""
        if slot:
            def memoized_fn(obj, *args):
                if hasattr(obj, slot):
                    return getattr(obj, slot)
                else:
                    val = fn(obj, *args)
                    setattr(obj, slot, val)
                    return val
        else:
            def memoized_fn(*args):
                if not memoized_fn.cache.has_key(args):
                    memoized_fn.cache[args] = fn(*args)
                return memoized_fn.cache[args]

            memoized_fn.cache = {}
        return memoized_fn
