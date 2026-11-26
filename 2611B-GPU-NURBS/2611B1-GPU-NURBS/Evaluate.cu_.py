
# TODO 2
# -exec source Evaluate.cu_.py Nrc

# -exec break Evaluate.cu:

# def regist(name)
# if name = knots:

def Nrc():
    knot = mem('knot', '', 20)
    print('knot', knot)

    print(', '.join(f'{name}({mem(name)})' for name in ('m', 'r', 'c', 'j', 'i', 'd')))

def NvT():
    knot = mem('NvT', '', 20)
    print('knot', knot)

    print(', '.join(f'{name}({mem(name)})' for name in ('m', 'r', 'c', 'j', 'i', 'd')))

def data():
    data = mem('data', '', 'kKnotCount')
    print('data', data)

