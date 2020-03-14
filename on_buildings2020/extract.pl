#!/usr/bin/perl
open d,"<raw.txt";
while(<d>){
    if($_ =~ /"iridoid"/ && $_ =~ /"from"/) {
        
    }
    print $_;

}