#保证训练图片的命令格式是name_id.jpg或者name_id.pgm。其中name相同表明是同一张人脸，
#id是此人的图像从0开始的编号，前后连贯。
import os, sys, re
from functools import cmp_to_key
def comp(e1, e2):
	(n1, i1) = re.split(r'_', e1)
	i1 = int(i1[0:-4])
	(n2, i2) = re.split(r'_', e2)
	i2 = int(i2[0:-4])
	if n1 != n2:
		if n1 > n2:
			return 1
		else:
			return -1
	return i1 - i2
def main():
	data = []
	root = sys.argv[1]
	for e in os.listdir(root):
		if e[-3:] == 'jpg' or e[-3:] == 'pgm':
			data.append(e)
	data.sort(key = cmp_to_key(comp))
	pname = ''
	index = 0
	for e in data:
		name = re.split(r'_', e)[0]
		suf = e[-4:]
		if name != pname:
			index = 0
			pname = name
		nname = name + '_' + str(index) + suf
		os.rename(os.path.join(root, e), os.path.join(root, nname))
		index += 1
		print('%s -> %s' % (e, nname))
if __name__ == '__main__':
	main()