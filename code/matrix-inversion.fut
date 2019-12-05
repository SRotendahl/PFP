-- Test matrix inversion
-- ==
-- input @ three_100_f32s
-- input @ three_1000_f32s
-- input @ three_10000_f32s
-- input @ three_100000_f32s
-- input @ three_1000000_f32s

-- let gaussian_elimination [n] [m] (A: [n][m]f32): [n][m]f32 =
--   loop A for i < n do
--     let v1 = A[0,i]
--     let elem k j = let x = unsafe (A[0,j] / v1) in
--                    if k < n-1  -- Ap case
--                    then unsafe (A[k+1,j] - A[k+1,i] * x)
--                    else x      -- irow case
--     in tabulate_2d n m elem

-- let matrix_inverse [n] (A: [n][n]f32): [n][n]f32 =
--   let zeros = replicate n (replicate n 0f32)
--   let M = map(\i -> let I = map(\j -> if i == j then 1f32 else zeros[i,j]) (iota n)
--                     in (A[i,:] ++ I)) (iota n)

--   in unsafe (gaussian_elimination M)[:,n:2*n]

-- let main [k][n][m] (A: [k][n][m]f32): [][][]f32 = map(\el -> matrix_inverse el) A

let gaussian_elimination [n] [m] (A: [n][m]f32): [n][m]f32 =
  loop A for i < n do
    let v1 = A[0,i]
    let elem k j = let x = unsafe (A[0,j] / v1) in
                   if k < n-1  -- Ap case
                   then unsafe (A[k+1,j] - A[k+1,i] * x)
                   else x      -- irow case
    in tabulate_2d n m elem

let matrix_inverse [n] (A: [n][n]f32): [n][n]f32 =
  let idMat = map2 (\x i -> let x' = copy x
  			    in update x' i 1)
		   (replicate n (replicate n 0)) (iota n)
  let gaus = map2 (++) A idMat |> gaussian_elimination
  in map (drop n) gaus

let main (xs: [][][]f32) : [][][]f32 =
  map (\x -> matrix_inverse x) xs

-- 1000000 5000000 10000000
