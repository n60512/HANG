WITH rand_train_set AS ( 
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase_maxlen30 
	ORDER BY RAND() 
	LIMIT 6080 
	) 
SELECT clothing_interaction6_itembase_maxlen30.rank, clothing_interaction6_itembase_maxlen30.ID, clothing_interaction6_itembase_maxlen30.reviewerID, clothing_interaction6_itembase_maxlen30.`asin`, 
clothing_interaction6_itembase_maxlen30.overall, clothing_interaction6_rm_sw.reviewText, clothing_interaction6_itembase_maxlen30.unixReviewTime 
FROM clothing_interaction6_itembase_maxlen30, clothing_interaction6_rm_sw 
WHERE clothing_interaction6_itembase_maxlen30.`asin` IN ( 
	SELECT * FROM rand_train_set 
) 
AND clothing_interaction6_itembase_maxlen30.ID = clothing_interaction6_rm_sw.ID 
ORDER BY `asin`,rank ASC 
-- LIMIT 10000
;
WITH rand_train_set AS ( 
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase_maxlen30 
	ORDER BY RAND() 
	LIMIT 6080 
	) 
, tmptable AS ( 
	SELECT DISTINCT(`asin`) 
	FROM clothing_interaction6_itembase_maxlen30 
	WHERE `asin` NOT IN ( 
		SELECT * FROM rand_train_set 
		) 
	-- LIMIT 1520 
	LIMIT 800 
	) 
SELECT clothing_interaction6_itembase_maxlen30.rank, clothing_interaction6_itembase_maxlen30.ID, clothing_interaction6_itembase_maxlen30.reviewerID, clothing_interaction6_itembase_maxlen30.`asin`, 
clothing_interaction6_itembase_maxlen30.overall, clothing_interaction6_rm_sw.reviewText, clothing_interaction6_itembase_maxlen30.unixReviewTime 
FROM clothing_interaction6_itembase_maxlen30, clothing_interaction6_rm_sw 
WHERE clothing_interaction6_itembase_maxlen30.`asin` IN (SELECT * FROM tmptable) 
AND clothing_interaction6_itembase_maxlen30.ID = clothing_interaction6_rm_sw.ID 
ORDER BY `asin`,rank ASC ;
;
-- 6080:1520 = 4:1
-- 6080:800 = 7.6:1