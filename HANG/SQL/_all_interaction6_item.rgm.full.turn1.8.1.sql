-- batch 64 (Full train PRM:1.033, RGM:repeat) 0702
SELECT * FROM _all_interaction6_item 
ORDER BY _turn, `asin`, rank
LIMIT 464640
;
SELECT * FROM _all_interaction6_item 
WHERE _no> 464640
AND _no <= 522624
-- 57984
ORDER BY _turn, `asin`, rank
;