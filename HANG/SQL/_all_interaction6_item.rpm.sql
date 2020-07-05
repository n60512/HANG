-- 0701
SELECT * FROM _all_interaction6_item 
ORDER BY _turn, `asin`, rank
LIMIT 581142
;
SELECT * FROM _all_interaction6_item 
WHERE _no> 581142
AND _no <= 593250
ORDER BY _turn, `asin`, rank
;
