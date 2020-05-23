WITH rand_train_set AS (
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase
	ORDER BY RAND() 
	-- LIMIT 15040 
	LIMIT 12800 
	) 
SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
clothing_interaction6_itembase.overall, clothing_interaction6_itembase.reviewText, clothing_interaction6_itembase.unixReviewTime
FROM clothing_interaction6_itembase, clothing_interaction6_rm_sw 
WHERE clothing_interaction6_itembase.`asin` IN ( 
	SELECT * FROM rand_train_set 
) 
AND clothing_interaction6_itembase.ID = clothing_interaction6_rm_sw.ID 
ORDER BY `asin`,rank ASC 
;
WITH rand_train_set AS (
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase
	ORDER BY RAND() 
	-- LIMIT 15040 
	LIMIT 12800 
	) 
, test_set AS (
	SELECT DISTINCT(`asin`)
	FROM clothing_interaction6_itembase
	WHERE `asin` NOT IN (
		SELECT * FROM rand_train_set 
		) 
	-- LIMIT 2000 
	LIMIT 1600 
	) 
SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
clothing_interaction6_itembase.overall, clothing_interaction6_itembase.reviewText, clothing_interaction6_itembase.unixReviewTime
FROM clothing_interaction6_itembase, clothing_interaction6_rm_sw 
WHERE clothing_interaction6_itembase.`asin` IN ( 
	SELECT * FROM test_set 
) 
AND clothing_interaction6_itembase.ID = clothing_interaction6_rm_sw.ID 
ORDER BY `asin`,rank ASC ;
;