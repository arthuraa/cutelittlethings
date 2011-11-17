

open Batteries_uni 
open Set 
(* #remove_printer Batteries_print.string_set;; *)

(* let stringSet = set_of_file "vocab.txt";; *)


module String = struct 
  include String   
  let for_all (p : char->bool) str = 
    str |> String.enum |> Enum.for_all p 
end 

let vocab_from_file file  = file |> File.lines_of |>  Array.of_enum

let vocab = vocab_from_file "vocab.txt" 
(* Array.findi (fun x-> x = "GOOD") vocab;; *)
let stop = vocab_from_file "stopword.txt"




(** calcuate the stop_index *)
let stop_index = Array.filter_map
  (fun x -> Array.Exceptionless.findi (fun y ->  y= x) vocab ) stop
  |> Array.map (fun x -> x + 1)
(** write the stop_index to index files *)
let _ = 
  stop_index |> Array.enum |> Enum.map (Int.to_string) |> File.write_lines "stop_index.txt" ;;


(** 1 - 11*)
let test_categories = 
  File.lines_of "test.category.txt" |> Enum.map Int.of_string |> IntSet.of_enum

(** 12 - 21*)
let train_categories = 
  File.lines_of "train.category.txt" |> Enum.map Int.of_string |> IntSet.of_enum
(* let tbl = Hashtbl.of_enum ([1,9;1,3;2,4;2,3]|>List.enum)  *)
(* in  Hashtbl.print Int.print Int.print stdout tbl;; *)

let train_reviewer = 
  File.lines_of "train.reviewer.txt" |> StringSet.of_enum 
(* - : int = 46184 *)


let test_reviewer = 
  File.lines_of "test.reviewer.txt" |> StringSet.of_enum 
(* - : int = 36975 *)




(* let _ =  *)
(*   stringSet  *)
(*   |> StringSet.enum  *)
(*   |> Enum.filter (String.for_all Char.is_uppercase ) *)
(*   |> Enum.count *)
 (* (Enum.print String.print stdout ) *)


(* StringSet.cardinal stringSet;; *)
(* - : int = 125166 *)
(* - : int =  11960 *)
