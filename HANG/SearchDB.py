from utils.DBconnector import DBConnection
import argparse



parser = argparse.ArgumentParser(description="Prog. for searching DB.")
parser.add_argument("--mode", default="title", choices=["title", "rating","both"],
                        help="" )

def searchProduct(conn, asin_):
    sql = "SELECT title FROM clothing_metadata WHERE `asin`='{}';".format(asin_)
    return conn.selection(sql)

def searchRating(conn, user_, asin_):
    sql = "SELECT overall AS rating FROM clothing_interaction6 WHERE `asin`='{}' AND reviewerID='{}';".format(asin_, user_)
    return conn.selection(sql)

if __name__ == "__main__":

    opt = parser.parse_args()
    print("""
        ######################################\n
        # This is prog. for searching DB,   #\n
        # you can exit by enter `exit`      #\n
        ######################################\n\n    
    """)


    conn = DBConnection()

    if opt.mode == 'title':
        # asin_ = "B00FPSX75W"
        while(True):
            asin_ = input('\nEnter productID you want to find: \n')

            if asin_ != 'exit':
                try:
                    res = searchProduct(conn, asin_)
                    print('Title:\n{}\n'.format(res[0]['title']))                
                except IndexError as msg:
                    print("`Data isn't exist!`")
            else:
                break
    
    elif opt.mode == 'rating':
        # asin_ = "B00FPSX75W"
        while(True):
            user_, asin_ = input('\nEnter reviewerID & productID you want to find: (slipt by ,)\n').replace(' ','').split(',')

            if asin_ != 'exit':
                try:
                    res = searchRating(conn, user_, asin_)
                    print('Rating:\n{}\n'.format(res[0]['rating']))                
                except IndexError as msg:
                    print("`Data isn't exist!`")
            else:
                break

    elif opt.mode == 'both':
        while(True):
            content = input('\nEnter reviewerID & productID you want to find: (slipt by ,)\n')

            if content != 'exit':
                user_, asin_ = content.replace(' ','').split(',')

                try:
                    res = searchRating(conn, user_, asin_)
                    print('Rating:\n{}\n'.format(res[0]['rating']))   

                    res = searchProduct(conn, asin_)
                    print('Title:\n{}\n'.format(res[0]['title']))  

                except IndexError as msg:
                    print("`Data isn't exist!`")
            else:
                break

    conn.close()    

    