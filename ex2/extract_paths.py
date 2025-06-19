import b

import re

def extractor(text:str):
    pattern = r"step \d+, chosen action: (\w), action: (\w)"
    match = re.search(pattern, text)

    return  match.group(1),match.group(2)

desired,actual = [],[]
for step in b.b:
    a = extractor(step[0])
    desired.append(a[0])
    actual.append(a[1])

print(desired)
print(actual)