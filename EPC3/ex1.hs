trapezoidFunc :: Float -> Float -> Float -> Float -> Float -> Float
trapezoidFunc a m n b x 
	| x < a = 0
	| x < m = (x-a)/(m-a)
    | x < n = 1
	| x < b = (b-x)/(b-n)
	| otherwise = 0

triangleFunc :: Float -> Float -> Float -> Float -> Float
triangleFunc a m b = trapezoidFunc a m m b


discretize :: Float -> Float -> Int -> [Float]
discretize min max num = take num [min, min+((max-min)/fromIntegral (num-1)) .. max]

activeSet :: Float -> Bool 
activeSet u = u > 0

alphaCut :: Float -> Float -> Bool
alphaCut a u = u >= a

complement :: Float -> Float
complement = ((-)1)

-- Ex 1.

funcA :: Float -> Float
funcA  = trapezoidFunc 0 0 5 15

funcB :: Float -> Float
funcB  = triangleFunc 5 15 25

funcC :: Float -> Float
funcC  = triangleFunc 15 25 35

funcD :: Float -> Float
funcD  = triangleFunc 25 35 45

funcE :: Float -> Float
funcE  = trapezoidFunc 35 45 50 50

discurseUniverse :: Int -> [Float]
discurseUniverse = discretize 0 50 

discurseUniverse50 :: [Float]
discurseUniverse50 = discurseUniverse 50

discurseUniverse1000 :: [Float]
discurseUniverse1000 = discurseUniverse 1000

discurseUniverse500 :: [Float]
discurseUniverse500 = discurseUniverse 500



-- 1. b)

--map funcA discurseUniverse1000
--map funcB discurseUniverse1000
--map funcC discurseUniverse1000
--map funcD discurseUniverse1000
--map funcE discurseUniverse1000

--1. c)

-- map (activeSet.funcA) discurseUniverse1000
-- map (activeSet.funcB) discurseUniverse1000
-- map (activeSet.funcC) discurseUniverse1000
-- map (activeSet.funcD) discurseUniverse1000
-- map (activeSet.funcE) discurseUniverse1000

--1. d)

-- funcA 
-- funcB
-- funcC
-- funcD
-- funcE

--1. e)

-- map ((alphaCut X).funcA) discurseUniverse1000
-- map ((alphaCut X).funcB) discurseUniverse1000
-- map ((alphaCut X).funcC) discurseUniverse1000
-- map ((alphaCut X).funcD) discurseUniverse1000
-- map ((alphaCut X).funcE) discurseUniverse1000

-- Ex2 

uA = map funcA discurseUniverse1000
uB = map funcB discurseUniverse1000
uC = map funcC discurseUniverse1000
uD = map funcD discurseUniverse1000
uE = map funcE discurseUniverse1000

op :: (Float -> Float -> Float) -> [Float]
op o = zipWith o (zipWith o (zipWith o (zipWith o uA uB) uC ) uD) uE

--2. b)

ex2b = op max

--2. c)

ex2c = op min

--2. d)

ex2d = map complement uC

--3. a)


-- main = print (map ((alphaCut 0.5).funcA) discurseUniverse)
