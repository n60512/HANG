WITH rand_train_set AS (
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase
	ORDER BY RAND() 
	LIMIT 15000
	-- LIMIT 5000
	) 
SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
clothing_interaction6_itembase.overall, clothing_interaction6_itembase.reviewText, clothing_interaction6_itembase.unixReviewTime
FROM clothing_interaction6_itembase
WHERE clothing_interaction6_itembase.`asin` IN ( 
	-- SELECT * FROM clothing_interaction6_usertrain 
	SELECT * FROM rand_train_set 
) 
ORDER BY `asin`,rank ASC 
-- LIMIT 10000
;
WITH rand_train_set AS (
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase
	ORDER BY RAND() 
	LIMIT 15000
	) 
, tmptable AS (
	SELECT DISTINCT(`asin`)
	FROM clothing_interaction6_itembase
	WHERE reviewerID NOT IN (
		-- SELECT * FROM clothing_interaction6_usertrain 
		SELECT * FROM rand_train_set 
		) 
	LIMIT 2500
	) 
SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
clothing_interaction6_itembase.overall, clothing_interaction6_itembase.reviewText, clothing_interaction6_itembase.unixReviewTime
FROM clothing_interaction6_itembase
WHERE clothing_interaction6_itembase.`asin` IN (SELECT * FROM tmptable) 
ORDER BY `asin`,rank ASC ;
;