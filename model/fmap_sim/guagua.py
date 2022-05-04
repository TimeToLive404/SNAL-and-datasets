print('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
print(f'ddd\N{SUPERSCRIPT THREE}')
print('111\u207b\u2077')
sup_map = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
formula = 'y=x2'
print(formula.translate(sup_map))
