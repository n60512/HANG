SELECT * FROM 
clothing_itembase_oversampling_v2_train
ORDER BY _turn,`asin`,rank ;
WITH rand_train_set AS (
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase
	ORDER BY RAND() 
	LIMIT 12800 
	) 
, validate_set AS (
	SELECT DISTINCT(`asin`)
	FROM clothing_interaction6_itembase
	WHERE `asin` NOT IN (
		SELECT * FROM rand_train_set 
		) 
	LIMIT 1600 
	) 
SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
clothing_interaction6_itembase.overall, clothing_interaction6_itembase.reviewText, clothing_interaction6_itembase.unixReviewTime
FROM clothing_interaction6_itembase 
WHERE clothing_interaction6_itembase.`asin` IN ( 
	SELECT * FROM validate_set 
) 
ORDER BY `asin`,rank ASC 
;