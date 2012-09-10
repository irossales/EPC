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

discurseUniverse :: [Float]
discurseUniverse = discretize 0 50 1000

-- 1. b)

--map funcA discurseUniverse
--map funcB discurseUniverse
--map funcC discurseUniverse
--map funcD discurseUniverse
--map funcE discurseUniverse

--1. c)

-- map (activeSet.funcA) discurseUniverse
-- map (activeSet.funcB) discurseUniverse
-- map (activeSet.funcC) discurseUniverse
-- map (activeSet.funcD) discurseUniverse
-- map (activeSet.funcE) discurseUniverse

--1. d)

-- funcA 
-- funcB
-- funcC
-- funcD
-- funcE

--1. e)

-- map ((alphaCut X).funcA) discurseUniverse
-- map ((alphaCut X).funcB) discurseUniverse
-- map ((alphaCut X).funcC) discurseUniverse
-- map ((alphaCut X).funcD) discurseUniverse
-- map ((alphaCut X).funcE) discurseUniverse
