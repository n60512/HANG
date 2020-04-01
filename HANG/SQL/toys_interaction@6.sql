SELECT *
FROM toys_interaction6 
WHERE reviewerID IN (SELECT * FROM toys_interaction6_usertrain) 
ORDER BY reviewerID,rank ASC 
-- LIMIT 10000
;
WITH tmptable AS (
	SELECT DISTINCT(reviewerID)
	FROM toys_interaction6
	WHERE reviewerID NOT IN (SELECT * FROM toys_interaction6_usertrain) 
	LIMIT 1000
	) 
SELECT *
FROM toys_interaction6 
WHERE reviewerID IN (SELECT * FROM tmptable) 
ORDER BY reviewerID,rank ASC ;
;