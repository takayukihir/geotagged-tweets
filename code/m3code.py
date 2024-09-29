import numpy as np
import geopy
import geopy.distance

DEGREE_IN_SEC = 60 * 60
BASIC_LAT_IN_SEC = 30
BASIC_LON_IN_SEC = 45
BASIC_LAT_IN_DEG = BASIC_LAT_IN_SEC / DEGREE_IN_SEC
BASIC_LON_IN_DEG = BASIC_LON_IN_SEC / DEGREE_IN_SEC

def code_to_grid_1d(primary, secondary, basic):
    return primary * 80 + secondary * 10 + basic

def grid_to_code_1d(coor):
	primary, pmod = divmod(coor, 80)
	secondary, basic = divmod(pmod, 10)
	return primary, secondary, basic

class m3code():
	'''
	Class for dealing with Basic Grid Square statistics.
	See https://www.stat.go.jp/english/data/mesh/index.html 
	for the details of the Grid Square statistics.
	'''
	def __init__(self, code):
		'''
		ycoor and xcoor are the grid coordinates 
		'''
		if len(code) != 8:
			raise ValueError('m3code must be 8-digit string')
		self.ycoor = code_to_grid_1d(int(code[0:2]), int(code[4]), int(code[6]))
		self.xcoor = code_to_grid_1d(int(code[2:4]), int(code[5]), int(code[7]))

	@classmethod
	def from_grid(cls, ycoor, xcoor):
		if not (isinstance(ycoor, int) and isinstance(xcoor, int)):
			raise TypeError('Initializing with non-integers')
		obj = cls.__new__(cls)
		obj.ycoor = ycoor
		obj.xcoor = xcoor
		return obj

	@classmethod
	def from_latlon(cls, lat, lon):
		'''
		initialize from geographic coordinates (latitude and longitude)
		'''
		obj = cls.__new__(cls)
		obj.ycoor = (lat * DEGREE_IN_SEC) // BASIC_LAT_IN_SEC
		obj.xcoor = ((lon - 100) * DEGREE_IN_SEC) // BASIC_LON_IN_SEC
		return obj

	def code(self):
		'''
		return the m3code as a string
		'''
		ycode = grid_to_code_1d(self.ycoor)
		xcode = grid_to_code_1d(self.xcoor)
		code_str = f'{ycode[0]:02}{xcode[0]:02}{ycode[1]}{xcode[1]}{ycode[2]}{xcode[2]}'
		return code_str

	def latlon(self):
		'''
		return latitude and longitude of the center point of the grid square
		'''
		return (self.ycoor + 0.5) * BASIC_LAT_IN_DEG, (self.xcoor + 0.5) * BASIC_LON_IN_DEG + 100

	def area(self):
		'''
		return the area
		'''
		southwest = (self.ycoor * BASIC_LAT_IN_DEG, self.xcoor * BASIC_LON_IN_DEG + 100)
		southeast = (self.ycoor * BASIC_LAT_IN_DEG, (self.xcoor + 1) * BASIC_LON_IN_DEG + 100)
		northwest = ((self.ycoor + 1) * BASIC_LAT_IN_DEG, self.xcoor * BASIC_LON_IN_DEG + 100)
		a = geopy.distance.distance(southwest, southeast).km * geopy.distance.distance(southwest, northwest).km
		return a

	def __add__(self, deltayx):
		'''
		add increments in grid coordinates, deltayx, to self.
		deltayx must be a pair of integers. 
		'''
		newy = self.ycoor + deltayx[0]
		newx = self.xcoor + deltayx[1]
		return m3code.from_grid(newy, newx)

	def __sub__(self, deltayx):
		return m3code + (- newy, - newx)

	# def lattice_neighborhood(self, dist):
	# 	if dist < 0:
	# 		raise ValueError('Negative distance is not allowed')
	# 	# dist == 0
	# 	elif dist < 1:
	# 		shell = [(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2) if (dy, dx) != (0, 0)]
	# 	# dist == 1
	# 	elif dist < np.sqrt(2):
	# 		shell = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) 
	# 		if (dy, dx) != (0, 0) and (abs(dy), abs(dx)) != (2, 2)]
	# 	# dist == np.sqrt(2)
	# 	elif dist < 2:
	# 		shell = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0)]
	# 	# dist == 2
	# 	elif dist < np.sqrt(5):
	# 		shell = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0)] \
	# 		+ [(dy, dx) for dy in range(-1, 2) for dx in [-3, 3]] \
	# 		+ [(dy, dx) for dy in [-3, 3] for dx in range(-1, 2)]
	# 	# dist == np.sqrt(5)
	# 	elif dist < np.sqrt(8):
	# 		shell = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) 
	# 		if (dy, dx) != (0, 0) and (abs(dy), abs(dx)) != (3, 3)]
	# 	# dist == np.sqrt(8)
	# 	elif dist < 3:
	# 		shell = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if (dy, dx) != (0, 0)]
	# 	# dist == 3
	# 	elif dist < np.sqrt(10):
	# 		shell = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if (dy, dx) != (0, 0)]\
	# 		+ [(dy, dx) for dy in range(-1, 2) for dx in [-4, 4]] \
	# 		+ [(dy, dx) for dy in [-4, 4] for dx in range(-1, 2)]
	# 	else:
	# 		raise ValueError('Shell is too large')

	# 	neigh = [(self + delta).code() for delta in shell]
	# 	return neigh

	def lattice_neighborhood(self, dist):
		if dist < 0:
			raise ValueError('Negative distance is not allowed')
		idist = int(np.ceil(dist))
		shell = [(dy, dx) for dy in range(-idist, idist+1) for dx in range(-idist, idist+1) 
			if (dy, dx) != (0, 0)]
		neigh = [(self + delta).code() for delta in shell]
		return neigh

	def geo_neighborhood(self, dist):
		if dist < 0:
			raise ValueError('Negative distance is not allowed')

		lat, lon = self.latlon()
		origin = geopy.Point(latitude=lat, longitude=lon)
		dist_obj = geopy.distance.great_circle(kilometers=dist)

		elat = []
		for bearing in [180, 0]:
			end_point = dist_obj.destination(point=origin, bearing=bearing)
			delta = ((end_point.latitude - lat) * DEGREE_IN_SEC) // BASIC_LAT_IN_SEC
			elat.append(int(delta) + 1)

		elon = []
		for bearing in [270, 90]:
			end_point = dist_obj.destination(point=origin, bearing=bearing)
			delta = ((end_point.longitude - lon) * DEGREE_IN_SEC) // BASIC_LON_IN_SEC
			elon.append(int(delta) + 1)

		shell = [(dy, dx) for dy in range(elat[0], elat[1]) for dx in range(elon[0], elon[1]) 
			if (dy, dx) != (0, 0)]

		neigh = [(self + delta).code() for delta in shell
			if geopy.distance.great_circle((lat, lon), (self + delta).latlon()).km < dist]

		return neigh

def m3code_property(row, property):
	'''
	property should be a m3code method function m3code.latlon or m3code.area
	'''
	code = m3code(row['m3code'])
	return property(code)

def latlon(df):
	return df.apply(lambda x:m3code_property(x, m3code.latlon), axis=1, result_type='expand')

def area(df):
	return df.apply(lambda x:m3code_property(x, m3code.area), axis=1)
