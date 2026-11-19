def Nrc():
    knot = mem('knot', '', 20)
    print('knot', knot)

    for name in ('m', 'r', 'c', 'j', 'i', 'd'):
        print(name, mem(name))
