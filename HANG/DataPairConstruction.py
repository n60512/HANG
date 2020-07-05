# code that build up all interaction data pair
from utils.DBconnector import DBConnection


text1 = (
"""
CREATE TEMPORARY TABLE _all_interaction AS 
WITH _asin_list AS ( 
	SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase 
	WHERE rank = &end& 
	) 
SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
clothing_interaction6_itembase.overall, clothing_interaction6_itembase.reviewText, clothing_interaction6_itembase.unixReviewTime 
FROM clothing_interaction6_itembase 
WHERE clothing_interaction6_itembase.`asin` IN ( 
	SELECT * FROM _asin_list 
) 
AND clothing_interaction6_itembase.rank >= &start& 
AND clothing_interaction6_itembase.rank <= &end& 
ORDER BY `asin`,rank ASC 
;
"""
)

text2 = (
"""
ALTER TABLE `_all_interaction` 
	ADD COLUMN `_turn` INT(1) NOT NULL DEFAULT '&start&' AFTER `unixReviewTime`;	
""")
text3 = (
"""
SELECT rank-&start& AS rank, ID, reviewerID, `asin`, overall, reviewText, unixReviewTime, `_turn` 
FROM _all_interaction 
ORDER BY `_turn`,`asin`,rank ASC 
;
"""
)
text4 = (
"""
DROP TABLE _all_interaction;
"""
)

insertsql = (
"""
INSERT INTO _all_interaction6_item_2 
SELECT rank-&start&+1, ID, reviewerID, `asin`, overall, reviewText, unixReviewTime, `_turn` 
FROM _all_interaction 
ORDER BY `_turn`,`asin`,rank ASC 
;
"""
)


if __name__ == "__main__":
    
    conn = DBConnection()
    t_len = 0
    for index in range(2,20):
        
        start = str(index)
        end = str(index+5)

        conn.execution(
            # text1.replace('&start&', '1').replace('&end&', '6')
            text1.replace('&start&', start).replace('&end&', end)
            )
        # conn.execution(text2.replace('&start&', '1'))
        conn.execution(text2.replace('&start&', start))
        # res = conn.selection(text3.replace('&start&', start)) 

        ## insert
        conn.execution(insertsql.replace('&start&', start))
        
        conn.execution(text4)      

        # t_len += len(res)
        # print('_turn:{}\t_len:{}\ttotal_len:{}'.format(index, len(res), t_len))  
        stop = 1
        pass

    conn.close()
    


    pass