WITH rand_train_set AS (
	SELECT DISTINCT(`asin`) FROM elec_interaction6_itembase
	ORDER BY RAND(42) 
	LIMIT 15040 
	) 
SELECT elec_interaction6_itembase.rank, elec_interaction6_itembase.ID, elec_interaction6_itembase.reviewerID, elec_interaction6_itembase.`asin`, 
elec_interaction6_itembase.overall, elec_interaction6_itembase.reviewText, elec_interaction6_itembase.unixReviewTime
FROM elec_interaction6_itembase 
WHERE elec_interaction6_itembase.`asin` IN ( 
	SELECT * FROM rand_train_set 
) 
ORDER BY `asin`,rank ASC 
-- LIMIT 10000
;
WITH rand_train_set AS (
	SELECT DISTINCT(`asin`) FROM elec_interaction6_itembase
	ORDER BY RAND(42) 
	LIMIT 15040 
	) 
, tmptable AS (
	SELECT DISTINCT(`asin`)
	FROM elec_interaction6_itembase
	WHERE `asin` NOT IN (
		SELECT * FROM rand_train_set 
		) 
	LIMIT 2000 
	) 
SELECT elec_interaction6_itembase.rank, elec_interaction6_itembase.ID, elec_interaction6_itembase.reviewerID, elec_interaction6_itembase.`asin`, 
elec_interaction6_itembase.overall, elec_interaction6_itembase.reviewText, elec_interaction6_itembase.unixReviewTime
FROM elec_interaction6_itembase 
WHERE elec_interaction6_itembase.`asin` IN (SELECT * FROM tmptable) 
ORDER BY `asin`,rank ASC ;
;