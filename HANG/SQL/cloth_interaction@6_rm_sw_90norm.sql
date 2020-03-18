WITH norm_table as (
	SELECT ID FROM clothing_interaction6_sen_len
	WHERE sentence_length <79
	)
, rand_train_set AS (
	SELECT reviewerID FROM clothing_interaction6
	WHERE ID IN (SELECT ID FROM norm_table)
	GROUP BY reviewerID
	HAVING COUNT(reviewerID) >=6
	ORDER BY RAND() 
	LIMIT 16000
	)
SELECT clothing_interaction6.rank, clothing_interaction6.ID, clothing_interaction6.reviewerID, clothing_interaction6.`asin`, 
clothing_interaction6.overall, clothing_interaction6_rm_sw.reviewText, clothing_interaction6.unixReviewTime
FROM clothing_interaction6, clothing_interaction6_rm_sw
WHERE clothing_interaction6.reviewerID IN ( 
	-- SELECT * FROM clothing_interaction6_usertrain 
	SELECT reviewerID FROM rand_train_set 
) 
AND clothing_interaction6.ID = clothing_interaction6_rm_sw.ID
ORDER BY reviewerID,rank ASC 
-- LIMIT 10000
;
WITH norm_table as (
	SELECT ID FROM clothing_interaction6_sen_len
	WHERE sentence_length <79
	)
, rand_train_set AS (
	SELECT reviewerID FROM clothing_interaction6
	WHERE ID IN (SELECT ID FROM norm_table)
	GROUP BY reviewerID
	HAVING COUNT(reviewerID) >=6
	ORDER BY RAND() 
	LIMIT 16000
	)
, tmptable AS (
	SELECT DISTINCT(reviewerID)
	FROM clothing_interaction6
	WHERE reviewerID NOT IN (
		-- SELECT * FROM clothing_interaction6_usertrain 
		SELECT reviewerID FROM rand_train_set 
		) 
	LIMIT 2000
	) 
SELECT clothing_interaction6.rank, clothing_interaction6.ID, clothing_interaction6.reviewerID, clothing_interaction6.`asin`, 
clothing_interaction6.overall, clothing_interaction6_rm_sw.reviewText, clothing_interaction6.unixReviewTime
FROM clothing_interaction6, clothing_interaction6_rm_sw
WHERE clothing_interaction6.reviewerID IN (SELECT * FROM tmptable) 
AND clothing_interaction6.ID = clothing_interaction6_rm_sw.ID
ORDER BY reviewerID,rank ASC ;
;