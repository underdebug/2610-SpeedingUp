
import gdb

expression = 'knots[mid]'
value = gdb.parse_and_eval(expression)
print(f'{expression} = {value}')


expression = 'knots[mid + 1]'
value = gdb.parse_and_eval(expression)
print(f'{expression} = {value}')

    # value = mem('knots[mid]')
    # print(f'{value}')
