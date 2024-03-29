-- Flat implementation of:
-- 1. partition2: reorders the elements of an array such that the ones
--                that succeed under a predicate comes before the ones
--                that fail (the predicate), but the relative order inside
--                the two classes is preserved
-- 2.partition2L: is the segmented/lifted version of `partition2`, i.e., it
--                semantically operates on an array of arrays and applies
--                `partition2` on each subarray (segment).
-- 3. quicksort:  is the flat-parallel version of quicksort algorithm.
--                quicksort implementation uses `partition2L`.
-- ==
-- compiled input { [29.0f32, 5.0f32, 7.0f32, 11.0f32, 2.0f32, 3.0f32, 13.0f32, 23.0f32, 17.0f32, 19.0f32] }
-- output { [2.0f32, 3.0f32, 5.0f32, 7.0f32, 11.0f32, 13.0f32, 17.0f32, 19.0f32, 23.0f32, 29.0f32] }
-- input @ Task4Data

---------------------
--- SgmSumInt     ---
---------------------
-- 2. sgmSumInt on integers, i.e., sgmIncScan (+) 0
let sgmSumInt [n] (flg : [n]i32) (arr : [n]i32) : [n]i32 =
  let flgs_vals =
    scan ( \ (f1, x1) (f2,x2) ->
            let f = f1 | f2 in
            if f2 > 0 then (f, x2)
            else (f, x1 + x2) )
         (0,0) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals

---------------------
--- MkFlags Array ---
---------------------

let mkFlagArray 't [m]
            (aoa_shp: [m]i32) (zero: t)       --aoa_shp=[0,3,1,0,4,2,0]
            (aoa_val: [m]t  ) : []t = unsafe  --aoa_val=[1,1,1,1,1,1,1]
  let shp_rot = map (\i->if i==0 then 0       --shp_rot=[0,0,3,1,0,4,2]
                         else aoa_shp[i-1]
                    ) (iota m)
  let shp_scn = scan (+) 0 shp_rot            --shp_scn=[0,0,3,4,4,8,10]
  let aoa_len = shp_scn[m-1]+aoa_shp[m-1]     --aoa_len= 10
  let shp_ind = map2 (\shp ind ->             --shp_ind=
                       if shp==0 then -1      --  [-1,0,3,-1,4,8,-1]
                       else ind               --scatter
                     ) aoa_shp shp_scn        --   [0,0,0,0,0,0,0,0,0,0]
  in scatter (replicate aoa_len zero)         --   [-1,0,3,-1,4,8,-1]
             shp_ind aoa_val                  --   [1,1,1,1,1,1,1]
                                              -- res = [1,0,0,1,1,0,0,0,1,0]

-----------------------
--- Parallel Filter ---
-----------------------
let partition2 [n] 't (conds: [n]bool) (dummy: t) (arr: [n]t) : (i32, [n]t) =
  let tflgs = map (\ c -> if c then 1 else 0) conds
  let fflgs = map (\ b -> 1 - b) tflgs

  let indsT = scan (+) 0 tflgs
  let tmp   = scan (+) 0 fflgs
  let lst   = if n > 0 then indsT[n-1] else -1
  let indsF = map (+lst) tmp

  let inds  = map3 (\ c indT indF -> if c then indT-1 else indF-1) conds indsT indsF

  let fltarr= scatter (replicate n dummy) inds arr
  in  (lst, fltarr)

-----------------------------------------
--- Weekly 2, TASK 4                  ---
--- The Lifted Version of Partition2  ---
--- (segmented version of Partition2) ---
-----------------------------------------
-- Please implement the function below, which is supposed to
--   be the lifted version of `partition2` function given above.
-- The current `main` function is testing quicksort, which will likely
--   not terminate (infinite loop) unless your implementation of `partition2L`
--   is correct. So, for debugging purposes write a new `main` function and
--   input dataset, which tests `partition2L`.
--
-- Arguments of `partition2L` are:
--   `(shp: [m]i32, arr: [n]t)` is the flat-representation of
--            the irregular 2-dim (input) array to be partitioned;
--            `shp` is its shape, and `arr` is its flat data;
--   `condsL` is an irregular 2-dim array of booleans, which has
--            the same shape (`shp`) and flat-length (`n`) as the
--            input to-be-partitioned array.
-- The result is a tuple:
--    the first element is an array of split points of size `m`,
--       i.e., the index in each segment where the `false` elements
--       start.
--    the second element is the flat-representation of the partitioned result:
--       the first element should simply be `shp` (redundant)
--       the second element should be the flat-data of the partitioned result.
-- Please note that `partition2` ends with a call to `scatter`, hence you will
--   probably need to apply the flattening rule you wrote to solve TASK3.
--
let sgmscan 't [n] (op: t->t->t) (ne: t) (flg : [n]i32) (arr : [n]t) : [n]t =
  let flgs_vals =
    scan ( \ (f1, x1) (f2,x2) ->
            let f = f1 | f2 in
            if f2 != 0 then (f, x2)
            else (f, op x1 x2) )
         (0,ne) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals

let scanExc 't [n] (op: t->t->t) (ne: t) (arr : [n]t) : [n]t =
    scan op ne <| map (\i -> if i>0 then unsafe arr[i-1] else ne) (iota n)

let segscatter 't (Sx: []i32, Dx: *[]t)
               (Di: []i32, Dv: []t)
               (shp: []i32) : ([]i32, *[]t) =
  let B = scanExc (+) 0 Sx
  let F = mkFlagArray shp 0 (1...(length shp))
  let IIi = sgmscan (+) 0 F F |> map (\i -> i-1)
  let BII = map (\i -> B[i]) IIi
  let absIndex = map2 (+) BII Di
  let sct = scatter Dx absIndex Dv
  in (shp, sct)

let partition2L 't [n] [m]
                -- the shape of condsL is also shp
                (condsL: [n]bool) (dummy: t)
                (shp: [m]i32, arr: [n]t) :
                ([m]i32, ([m]i32, [n]t)) =
  let begs   = scan (+) 0 shp
  let flags  = mkFlagArray shp 0i32 (map (+1) (iota m))
  let outinds= sgmSumInt flags <| map (\f -> if f==0 then 0 else f-1) flags
  let tflags = map (\ c -> if c then 1 else 0) condsL
  let fflags = map (\ b -> 1 -b) tflags

  let indsT = sgmSumInt flags tflags
  let tmp = sgmSumInt flags fflags
  let lst = map (\b -> if n > 0 then indsT[b-1] else -1) begs
  let indsF = map2 (\t o -> t + lst[o]) tmp outinds
  let inds = map3 (\c indT indF -> if c then indT-1 else indF-1) condsL indsT indsF
  let fltarr = segscatter (shp, replicate n dummy) (inds, arr) shp
  in  (shp, fltarr)

-----------------------
--- Flat Quicksort
-----------------------
let randomInd ( (lb,ub) : (i32,i32) ) (count : i32) : i32 =
  if lb > ub then 0 else
  ( (count+lb+ub) % (ub - lb + 1) ) + lb

let isSorted [n] (arr: [n]f32) : bool =
    map (\i-> unsafe (arr[i] <= arr[i+1])) (iota (n-1))
    |> reduce (&&) true

let quicksortL [n][m] (shp: [m]i32, arr: [n]f32) : ([]i32, []f32) =
  let stop  = isSorted arr
  let count = 0

  let (shp,arr,_,_) =
    loop(shp,arr,stop,count) while (!stop) do
      let begs   = scan (+) 0 shp
      let flags  = mkFlagArray shp 0i32 <| map (+1) <| iota (length shp)

      let outinds= sgmSumInt flags <| map (\f -> if f==0 then 0 else f-1) flags

      let rL   = map (\u -> randomInd(0,u-1) count) shp
      let pivL = map3(\r l i -> if l <= 0 then 0.0
                                else let off = if i > 0 then unsafe begs[i-1] else 0
                                     in  unsafe arr[off + r]
                     ) rL shp (iota (length shp))

      let condsL = map2(\a sgmind -> unsafe pivL[sgmind] > a ) arr outinds

      let (ps, (_,arr')) = partition2L condsL 0.0f32 (shp, arr)

      -- shp' = [p, n-p]
      let shp' = filter (!=0) <| flatten <| map2 (\p s -> if s==0 then [0,0] else [p,s-p]) ps shp

      let stop' = isSorted arr'
      in (shp', arr', stop', count+1)
  in (shp,arr)

-----------------------
---   test program  ---
-----------------------

--let main [n] (arr: [n]i32) : (i32, [n]i32) =
--    partition2 (map (\x -> (x % 2) == 0i32) arr) 0i32 arr

let main0 [m][n] (shp: [m]i32) (arr: [n]i32) : ([m]i32, [m]i32, [n]i32) =
    let (ps, (shp',arr')) = partition2L (map (\x -> (x % 2) == 0i32) arr) 0i32 (shp, arr)
    in  (ps, shp', arr')

-- futhark dataset -b --f32-bounds=-1000000.0:1000000.0 -g [10000000]f32 | ./quicksort-flat -t /dev/stderr -r 2 > /dev/null
let main [n] (arr: [n]f32) =
    unsafe
    let (_,res) = quicksortL ([n], arr)
    in  res
