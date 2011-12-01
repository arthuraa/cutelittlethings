scp ./submit.txt hongboz@eniac.seas.upenn.edu:./final/

sudo ssh -L 127.0.0.2:139:smb.seas.upenn.edu:139 hongboz@seas.upenn.edu

cd final

turnin -c cis520 -p leaderboard submit.txt