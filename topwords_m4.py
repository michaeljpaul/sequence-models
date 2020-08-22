import sys
import math
from operator import itemgetter

filename = sys.argv[1]

follows = {}
followed = {}

infile = file(filename+".lambda", "r")
z = 0
for line in infile:
	tokens = line.strip().split()

	follows[z] = {}
	z0 = 0
	for p in tokens:
		p = float(p)
		if math.fabs(p) > 0.1:  # only print weights > 0.1
			follows[z][z0] = p
			if z0 not in followed: followed[z0] = {}
			followed[z0][z] = p
		z0 += 1
	z += 1
infile.close()

infile = file(filename+".assign", "r")

Z = 0
count = {}
countB = {}

for line in infile:
	tokens = line.split()
	if len(tokens) < 2: continue

	garbage = tokens.pop(0)
	garbage = tokens.pop(0)
	id = tokens.pop(0)

	for token in tokens:
		parts = token.split(":")
		x = int(parts.pop())
		z = int(parts.pop())
		word = ":".join(parts)
 
		if x == 1:
			if z not in count:
				count[z] = {}
			if word not in count[z]:
				count[z][word] = 0
			count[z][word] += 1

			if z > Z: Z = z
		else:
			if word not in countB:
				countB[word] = 0
			countB[word] += 1
infile.close()
Z += 1

print("Background\n")
w = 0
words = sorted(countB.items(), key=itemgetter(1), reverse=True)
for word, v in words:
	print(word)
	w += 1
	if w >= 50: break
print("\n")

for z in range(Z):
	print("Topic %d\n" % (z+0))

	w = 0
	words = sorted(count[z].items(), key=itemgetter(1), reverse=True)
	for word, v in words:
		print(word)
		w += 1
		if w >= 50: break
	print("\n")

	print("Follows:")
	if z not in follows: follows[z] = {}
	for z0 in follows[z]:
		print(z0, follows[z][z0])
	print("Followed by:")
	if z not in followed: followed[z] = {}
	for z0 in followed[z]:
		print(z0, followed[z][z0])
	print("\n")

