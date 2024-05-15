import React, { useState } from "react";
import Modal from "./Modal";
const Card = ({ book }) => {
  const [show,setShow]= useState(false);
  const [bookItem,setItem]= useState();
  return (
    <>
      {book.map((item) => {
        let thumbnail=item.volumeInfo.imageLinks && item.volumeInfo.imageLinks.smallThumbnail;
        let amount=item.saleInfo.listPrice && item.saleInfo.listPrice.amount;
         if (thumbnail !== undefined /*&& amount !== undefined*/ )
        {
            return (
                <>
                <div className="card" onClick={()=>{setShow(true);setItem(item)}}>
                  <img src={thumbnail} alt="java book" />
                  <div className="bottom">
                    <h3 className="title">{item.volumeInfo.title}</h3>
                    <p className="amount">&#36;NOT FOR SALE{amount}</p>
                  </div>
                </div>
                <Modal show={show} item={bookItem} onclose={()=>setShow(false)}/>
                </>
              )
        }
        
      })}
    </>
  )
}
export default Card;
