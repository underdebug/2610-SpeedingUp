
control_points = cpu('control_points', Size=int(gdb.parse_and_eval('cp_u')) * int(gdb.parse_and_eval('cp_v')))
print(control_points)
