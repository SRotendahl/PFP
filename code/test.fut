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

let segscatter (Sx: []i32, Dx: *[]i32)
               (Di: []i32, Dv: []i32)
               (shp: []i32) : ([]i32, *[]i32) =
  let B = scanExc (+) 0 Sx
  let F = mkFlagArray shp 0 (1...(length shp))
  let IIi = sgmscan (+) 0 F F |> map (\i -> i-1)
  let BII = map (\i -> B[i]) IIi
  let absIndex = map2 (+) BII Di
  let sct = scatter Dx absIndex Dv
  in (shp, sct)

let main : ([]i32, []i32) =
  unsafe
  segscatter ([3,2,4], [1,2,3,4,5,6,7,8,9]) ([1,2,0,3,1], [10,20,30,40,50]) [2,1,2]
