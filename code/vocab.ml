

open Batteries_uni 
open Set 
module String = struct 
  include String   
  let for_all (p : char->bool) str = 
    str |> String.enum |> Enum.for_all p 
end 

let vocab_from_file file  = file |> File.lines_of |>  Array.of_enum
let vocab = vocab_from_file "vocab.txt" 
(* Array.findi (fun x-> x = "GOOD") vocab;; *)

let vocab_hashbl = 
  vocab
  |> Array.enum 
  |> Enum.mapi (fun i s -> (s,i)) 
  |> Hashtbl.of_enum

open Camlp4.PreCast 
module MGram =MakeGram(Lexer) 
let hashtbl = Hashtbl.create 100 
let print_hashtbl = 
  Hashtbl.print String.print String.print stdout 

let candidates = MGram.Entry.mk "candidates" 
let idx = ref  0 
let _ = 
  EXTEND MGram 
    GLOBAL: candidates ; 
    candidates : 
      [[ "&" ;  x = my_word ; 
       INT ; INT ; ":"; cans = LIST1 my_word SEP "," -> 
       (* incr idx ;  *)
       (* print_string (vocab.(!idx)) ;  *)
       (* print_string " " ;  *)
       (* print_string x ;  *)
       (* print_newline (); *)

        ( try 
           let id = Hashtbl.find vocab_hashbl x  in 
           if String.ends_with x "n" && List.mem (x ^ "'t") cans then 
             Hashtbl.add  hashtbl id  (x^"'t") 
           else if String.ends_with x "nt"
           then 
             let  prob = (String.sub x 0 (String.length x - 2) ^ "n't")
             in if List.mem prob cans  then 
                 Hashtbl.add hashtbl id prob 
               else 
                 (try 
                    let e = List.find (fun s -> s <> "") cans in 
                    Hashtbl.add hashtbl id e 
                  with 
                      exn -> ())
           else 
             (try 
                let e =  List.find (fun s -> s <> "") cans in 
                Hashtbl.add hashtbl id  e  
              with 
                  exn ->  ())
         with Not_found -> () )
       | "*" ->  ()
         (* incr idx ;  *)
         (* print_string (vocab.(!idx)) ;  *)
         (* print_string " " ;  *)
         (* print_string  ;  *)
         (* print_newline () *)
         
       | "#" -> incr idx ; ()
       | "@" -> ()
       ]
      ]; 
    my_word : [[x = LIDENT -> x 
             |y = UIDENT -> y
             |v = SELF; "-"; u = SELF -> ""
             |u = SELF; v = SELF -> ""
                     ]]; 
    END 

let parse_candidates = try 
                         MGram.parse_string candidates (Loc.mk "<string>") 
  with exn -> raise exn 



let replace_word file_name =
  let _ = Hashtbl.clear hashtbl in 
  let count = ref 0 in 
  idx := 0 ; 
  file_name
  |> File.lines_of 
  |> Enum.iter (fun str -> incr count ; 
    if !count mod 10000 = 0 then (print_int !count; print_newline ());
    if str <> "" then 
      try parse_candidates str
      with exn -> begin 
        print_string ("parse error" ^ string_of_int !count  ); 
        raise exn
      end 
  )



let trim_nonchar s =
  let len = String.length s          in
  let rec aux_1 i = (*locate leading whitespaces*)
    if   i = len then None (*The whole string is whitespace*)
    else if not(Char.is_letter s.[i]) then aux_1 (i + 1)
    else Some i in
  match aux_1 0 with
    | None -> ""
    | Some last_leading_whitespace ->
  let rec aux_2 i =
    if   i < 0 then None(*?*)
    else if not(Char.is_letter s.[i]) then aux_2 (i - 1)
    else Some i in
  match aux_2 (len - 1) with
    | None -> ""
    | Some first_trailing_whitespace ->
	String.slice ~first:last_leading_whitespace 
          ~last:(first_trailing_whitespace + 1) s



let _ = replace_word "vocab.replace.txt" 
(* #remove_printer Batteries_print.string_set;; *)
(** hash_tbl is the initial  mapping fuction 
   0 -> "I" *)
let trans = String.lowercase |- trim_nonchar

(* let trans =  trim  *)

let string_set = 
  let _ = 
    Array.iteri (fun i v -> 
      if not(Hashtbl.mem hashtbl i)
      then 
        Hashtbl.add hashtbl i v
    ) vocab
  in Hashtbl.values hashtbl 
  |> Enum.filter_map 
      (fun s -> 
       let v = trans s  in 
       if v = "" then None
       else Some v ) (* ingore*)
  |> StringSet.of_enum

(** ignore case : 84838 -> 57782 -> 49938*)
(** co_domain 
   "i" -> 21257 *)  

let co_domain = 
  string_set 
  |> StringSet.enum 
  |> Enum.mapi (fun i v -> (v,i)) 
  |> Hashtbl.of_enum 


(** it finally appeared *)
let mapping = 
  Array.mapi (fun i v -> (i, 
                          let s = trans ( Hashtbl.find hashtbl i) in 
                          let ind = 
                            (try Hashtbl.find co_domain s
                            with
                                Not_found -> -1  ) 
                          in 
                          (ind, s))) vocab 
  |> Array.enum 
  |> Hashtbl.of_enum 

let mapping_vocab = 
  string_set 
  |> StringSet.enum 
  |> Enum.mapi (fun i v -> v )
  |> Array.of_enum 
;;

let id = 
  (Hashtbl.find mapping (Hashtbl.find vocab_hashbl "disapionted"),
   Hashtbl.find mapping (Hashtbl.find vocab_hashbl "disappointed"))

;; 
(* test

Hashtbl.find mapping 0 , vocab.(0)
Hashtbl.find mapping 1 , vocab.(1)
*)


let search str = 
  Array.findi (fun x -> x = str) vocab;;

(** first mapping *)
let write_mapping = 
  mapping 
  |> Hashtbl.enum
  |> Enum.map (fun (i,(j,v))  -> 
                    string_of_int (i+1) ^ "  " ^ string_of_int  (j+1))
  |>
      File.write_lines "mapping_to_correct_words.txt"




let _ = 
  mapping_vocab 
  |> Array.enum 
  |> File.write_lines "mapping_correct_words.txt"

;;

#u "./stem.py  > mapping_stemming.txt" ;;

let stemming = 
   "mapping_stemming.txt" (* generated by python *)
  |> File.lines_of
  |> Enum.mapi (fun i line -> 
    let (k,v) = String.split line " "
    in (String.trim k, String.trim v))
  |> Hashtbl.of_enum


let co_domain_stemming = 
  stemming |> Hashtbl.values 
  |> StringSet.of_enum  
  |> StringSet.enum 
  |> Enum.mapi (fun i v -> (v,i))
  |> Hashtbl.of_enum 

let co_domain_stemming_reverse = 
  stemming |> Hashtbl.values 
  |> StringSet.of_enum  
  |> StringSet.enum 
  |> Enum.mapi (fun i v -> (i,v))
  |> Hashtbl.of_enum 
  
exception Temp 

let mapping_2 = 
  Array.mapi (fun i v -> 
  (i, 
   (let j, correct_word = Hashtbl.find mapping i  in 
   (try    
     let stemming_word = 
       try 
         Hashtbl.find stemming correct_word 
       with Not_found -> raise Temp 
     in
     Hashtbl.find co_domain_stemming stemming_word 
   with Temp  -> -1) ))) vocab

let _ = 
  mapping_2 
  |> Array.enum 
  |> Enum.map (fun (i,j) -> 
    string_of_int (i + 1) ^ " " ^
    string_of_int (j + 1))
  |> File.write_lines "mapping_index_stemming.txt"

let _ = 
  co_domain_stemming_reverse 
  |> Hashtbl.values
  |> File.write_lines "mapping_stemming_words.txt"


let conform i = begin
 print_string vocab.(i); print_string " ";
 let _, j = mapping_2.(i) in (
 print_string (Hashtbl.find co_domain_stemming_reverse j); 
 print_string "\n"; )
end 
;;


let a = 
  let index = ref 0 in 
  "bigram_vocab.txt" 
  |> File.lines_of
  |> Enum.filter_map (fun s -> incr index ; 
    if String.starts_with s "not" then
      Some (string_of_int !index )
    else None)
  |> File.write_lines "bigram_not_index.txt"

(** seldom very ... *)
let b = 
  let index = ref 0 in 
  "bigram_vocab.txt" 
  |> File.lines_of
  |> Enum.filter_map (fun s -> incr index ; 
    if String.starts_with s "never" then
      Some (string_of_int !index )
    else None)
  |> File.write_lines "bigram_never_index.txt"

let filter_word word = 
  let index = ref 0 in 
  "bigram_vocab.txt" 
  |> File.lines_of
  |> Enum.filter_map (fun s -> incr index ; 
    if String.starts_with s word then
      Some (string_of_int !index )
    else None)
  |> File.write_lines ("bigram_" ^ word ^  "_index.txt") 

let _ = filter_word "not"
let _ = filter_word "very"
let _ = filter_word "seldom"
let _ = filter_word "hardly"

let _ =
  ["always"; "every"; "never"; "often"; "rarely"; "seldom";
   "usually"; "very"; "not"; "hardly"]
  |> List.iter filter_word 
   
;;

;;










(* mapping_vocab.(54273);;
- : Batteries_uni.Set.StringSet.elt = "disappointed"
*)


(*
Hashtbl.find vocab_hashtdisapointing
*)

(*
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


let xs = "dict.txt"
  |> File.lines_of 
  |> Enum.filter_map (fun line -> 
    let words = (SPLIT space+) line in 
    if float_of_string (List.nth words 1 ) > 0.99 then 
      Some (List.nth words 0)
    else None ) 

  |>Enum.filter_map (fun word -> try 
                             Some (string_of_int 
                                     (Array.findi (fun x -> x = word) vocab))
    with _ -> None
  ) 
  |> File.write_lines "dict_index.txt" 


  (* |> Enum.iter (fun i ->  *)
  (*   print_int i ; *)
  (*   print_string (vocab.(i)); *)
  (*   print_string "  ";  *)
  (*   print_newline ()) *)


let is_adj_adv_pred str = 
  (String.for_all 
     (fun c -> not (Char.is_digit c)) str)
  &&
    (String.length str > 2 )


let divide_by_prop file_name = 
  file_name 
  |> File.lines_of 
  |> Enum.filter_map (fun line -> 
    let words = (SPLIT space+) line in 
    let [name; noun; verb; adj; adv] = words in 
    let is_adj_adv = float_of_string adj +. float_of_string adv > 0.33 in 
    if is_adj_adv && is_adj_adv_pred name then
      let idx = Array.findi (fun x -> x = name)vocab in 
      Some (string_of_int (idx + 1)) 
    else 
      None
  ) |> File.write_lines "adj_adv_index.txt"


let junk  =  
  let c = ref 0 in 
  Array.filter_map  (fun str -> 
    c:= !c + 1 ; 
    if not  (String.for_all 
            (fun c -> not (Char.is_digit c)) str)
    then Some ((string_of_int !c ))
    else None
  ) vocab

(** write num digit words to file *)
let _ = 
  junk |> Array.enum |> File.write_lines "number_junk_words_index.txt"


let train_text file_name = 
  file_name 
  |> File.lines_of 
  |> Array.of_enum


(** TODO 
    Negative words will be very important 
    correct spelling error 
*)
let train_raw_text = train_text "train.text.txt" 

let _ = 
  let c = ref 0 in 
  train_raw_text |> 
      Array.filter_map 
      (fun text -> 
        incr c ; 
        if flip String.exists "NOT" text 
        then Some (string_of_int !c)
        else None
        )
  |> Array.enum |> File.write_lines "NOT_index.txt"



module ISHashtbl = Hashtbl.Make (struct 
  type t = string 
  let equal x  y = String.icompare  x y = 0 
  let hash = Hashtbl.hash 
end )

let table file_name = 
  file_name 
  |> File.lines_of 
  |> Enum.map (fun w -> String.trim w, 1)
  |> ISHashtbl.of_enum

let negative_table = "negative_words.txt" |> table 

let negative_list 
    = 
  let c = ref 0 
  in 
  Array.filter_map (fun w ->
    incr c ; 
    try 
      (ISHashtbl.find negative_table w )|>ignore; 
      Some (string_of_int !c)
    with Not_found -> None )
    vocab 
  |> Array.enum |> File.write_lines "negative_index.txt"

*)
(* let lst = MGram.Entry.mk "lst" *)

(* EXTEND MGram  *)
(*   GLOBAL: lst ;  *)
(*   lst: [ *)
(*     [ "["; lst = LIST1 [x = INT -> int_of_string x ] SEP ";";  "]" ->  *)
(*     Array.of_list lst  *)
(*     | v = INT -> [| int_of_string v |] ]                  *)
(*   ] ;  *)
(* END   *)

(* let lst_parse =   *)
(*   MGram.parse_string lst (Loc.mk "<string>")  *)

(* let train_raw_id file_name = file_name  *)
(*   |> File.lines_of  *)
(*   |> Enum.map (lst_parse) *)
(*   |> Array.of_enum *)

(* let train_word_id = train_raw_id "train.word_idx.txt"  *)

(* 457  lousy product
 *) 
(* let a  =  *)
(*   parse_candidates "& doesn 54 0: dozen, does, dowsing, Downs, downs, dosing, dozens, Dons, dens, dons, Deon, dose, Dean, dean, dies, doe's, doers, dossing, dousing, down, design, doses, doyen, dozes, DOS, Don, Dyson, den, doeskin, doesn't, don, dos, dosed, Dodson, Dotson, Donn, do's, doss, dues, toes, die's, doer's, dose's, doze's, Don's, don's, DOS's, due's, toe's, down's, Deon's, x-dozen's, den's, Donn's";; *)

