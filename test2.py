import CoolProp.CoolProp as CP
a = [1,2,3,4]

for i in range(len(a)-1, -1, -1):
    print(i, a[i-1])

print(CP.PropsSI("viscosity", "P", 101325, "H", 0, "n-Dodecane"))

for j in range(2):
    print(j)
