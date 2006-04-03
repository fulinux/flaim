//------------------------------------------------------------------------------
// Desc:	Data Vector
//
// Tabs:	3
//
//		Copyright (c) 2003-2006 Novell, Inc. All Rights Reserved.
//
//		This program is free software; you can redistribute it and/or
//		modify it under the terms of version 2 of the GNU General Public
//		License as published by the Free Software Foundation.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//		GNU General Public License for more details.
//
//		You should have received a copy of the GNU General Public License
//		along with this program; if not, contact Novell, Inc.
//
//		To contact Novell about this file by physical or electronic mail,
//		you may find current contact information at www.novell.com
//
// $Id: DataVector.java 3110 2006-01-19 13:09:08 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------
package xflaim;

/**
 * This class implements an interface to the XFlaim IF_DataVector class.
 */
public class DataVector
{
	long		m_this;
	DbSystem	m_dbSystem;

	/**
	 * Constructor for the DataVector object.  This object provides access to
	 * the XFlaim IF_DataVector interface.  All methods defined by the
	 * IF_DataVector interace are accessible through this Java object.
	 * 
	 * @param lRef A reference to a C++ IF_DataVector object
	 * @param dbSystem A reference to a DbSystem object
	 */
	public DataVector(
		long		lRef,
		DbSystem	dbSystem)
	{
		super();
		m_this = lRef;
		m_dbSystem = dbSystem;
	}
	
	/**
	 * Finalizer method, used to ensure that we release the actual C++ object.
	 */
	public void finalize()
	{
		if (m_this != 0)
		{
			_release( m_this);
			m_this = 0;
		}
		
		m_dbSystem = null;
	}
	
	/**
	 * Method to set the document Id of the search target.
	 * @param lDocId
	 */
	public void setDocumentID(
		long		lDocId)
	{
		_setDocumentId( m_this, lDocId);
	}
	
	/**
	 * Method to set the ID of the search target.  The ID referred to here is
	 * the node Id which is actually a 64 bit unsigned value in the XFlaim
	 * database. 
	 * @param iElementNumber
	 * @param lID
	 * @throws XFlaimException
	 */
	public void setId(
		int			iElementNumber,
		long			lID) throws XFlaimException
	{
		_setID( m_this, iElementNumber, lID);
	}
	
	/**
	 * Method to set the name ID of the search target.  the name Id is a
	 * numeric value that is used to represent the field or tag of the
	 * target element.
	 * @param iElementNumber
	 * @param iNameId
	 * @param bIsAttr A boolean flag that indicates whether or not the key 
	 * is an attribute.
	 * @param bIsData A boolean flag that indicates whether or not the key
	 * is a data component.
	 * @throws XFlaimException
	 */
	public void setNameId(
		int			iElementNumber,
		int			iNameId,
		boolean		bIsAttr,
		boolean		bIsData) throws XFlaimException
	{
		_setNameId( m_this, iElementNumber, iNameId, bIsAttr, bIsData);
	}

	/**
	 * Method to set the value of the target key to an integer value.  For 
	 * purposes of this interface, an integer is defined to be 32 bits, signed.
	 * The iNum parameter will be tested to ensure that is falls within range.
	 * If it is too large, an exception will be thrown.
	 * @param iElementNumber
	 * @param iNum The 32 bit signed integer value
	 * @throws XFlaimException
	 */
	public void setINT(
		int			iElementNumber,
		int			iNum) throws XFlaimException
	{
		_setINT( m_this, iElementNumber, iNum);
	}
	
	/**
	 * Special purpose function - NOT for general consumption.
	 * @param iElementNumber
	 * @param iUNum
	 * @throws XFlaimException
	 */
	public void setUINT(
		int			iElementNumber,
		int			iUNum) throws XFlaimException
	{
		_setUINT( m_this, iElementNumber, iUNum);
	}

	/**
	 * Method to set the value of the target key to a long value.  For 
	 * purposes of this interface, a long is defined to be 64 bits, signed.
	 * @param iElementNumber
	 * @param lNum The 64 bit signed integer value
	 * @throws XFlaimException
	 */
	public void setLong(
		int			iElementNumber,
		long			lNum) throws XFlaimException
	{
		_setLong( m_this, iElementNumber, lNum);
	}

	/**
	 * Method to set the value of the target key to a string value.
	 * @param iElementNumber
	 * @param sValue
	 * @throws XFlaimException
	 */
	public void setString(
		int			iElementNumber,
		String		sValue) throws XFlaimException
	{
		_setString( m_this, iElementNumber, sValue);
	}
	
	/**
	 * Method to set the value of the target key to a binary value.
	 * @param iElementNumber
	 * @param Value
	 * @throws XFlaimException
	 */
	public void setBinary(
		int			iElementNumber,
		byte[]		Value) throws XFlaimException
	{
		_setBinary( m_this, iElementNumber, Value);
	}
	
	/**
	 * Method to set a flag in the target key that indicates that the key is
	 * right truncated.
	 * @param iElementNumber
	 */
	public void setRightTruncated(
		int		iElementNumber)
	{
		_setRightTruncated( m_this, iElementNumber);
	}

	/**
	 * Method to set a flag in the target key that indicates that the key is
	 * left truncated.
	 * @param iElementNumber
	 */
	public void setLeftTruncated(
		int		iElementNumber)
	{
		_setLeftTruncated( m_this, iElementNumber);
	}

	/**
	 * Method to clear a flag in the target key that indicates that the key
	 * is right truncated.
	 * @param iElementNumber
	 */
	public void clearRightTruncated(
		int		iElementNumber)
	{
		_clearRightTruncated( m_this, iElementNumber);
	}

	/**
	 * Method to clear a flag in the target key that indicates that the key
	 * is left truncated.
	 * @param iElementNumber
	 */
	public void clearLeftTruncated(
		int		iElementNumber)
	{
		_clearLeftTruncated( m_this, iElementNumber);
	}

	/**
	 * Method to get the Document ID of the target key.
	 * @return Document Id
	 */
	public long getDocumentID()
	{
		return _getDocumentID( m_this);
	}

	/**
	 * Method to get the node Id of the element specified (iElementNumber) of
	 * the target key.
	 * @param iElementNumber
	 * @return Node Id
	 */
	public long getID(
		int		iElementNumber)
	{
		return _getID( m_this, iElementNumber);
	}

	/**
	 * Method to get the name Id of the element specified (iElementNumber) of
	 * the target key.
	 * @param iElementNumber
	 * @return Name Id
	 */
	public int getNameId(
		int		iElementNumber)
	{
		return _getNameId( m_this, iElementNumber);
	}

	/**
	 * Method to find out if the element specified (iElementNumber) is an
	 * attribute of the target key.
	 * @param iElementNumber
	 * @return boolean true or false
	 */
	public boolean isAttr(
		int		iElementNumber)
	{
		return _isAttr( m_this, iElementNumber);
	}

	/**
	 * Method to find out if the element specified (iElementNumber) is a data
	 * component of the target key.
	 * @param iElementNumber
	 * @return boolean true or false
	 */
	public boolean isDataComponent(
		int		iElementNumber)
	{
		return _isDataComponent( m_this, iElementNumber);
	}

	/**
	 * Method to find out if the element specified (iElementNumber) is a key
	 * component of the target key.
	 * @param iElementNumber
	 * @return boolean true or false
	 */
	public boolean isKeyComponent(
		int		iElementNumber)
	{
		return _isKeyComponent( m_this, iElementNumber);
	}

	/**
	 * Method to get the length of the data value of the element specified 
	 * (iElementNumber) of the target key.
	 * @param iElementNumber
	 * @return The data length
	 */
	public int getDataLength(
		int		iElementNumber)
	{
		return _getDataLength( m_this, iElementNumber);
	}

	/**
	 * Desc:
	 */
	public int getDataType(
		int		iElementNumber)
	{
		return _getDataType( m_this, iElementNumber);
	}

	/**
	 * Method to get the value of the element specified (iElementNumber) of the
	 * target key as an integer.  An integer is a 32 bit signed value.
	 * @param iElementNumber
	 * @return 32 bit signed integer
	 * @throws XFlaimException
	 */
	public int getINT(
		int		iElementNumber) throws XFlaimException
	{
		return _getINT( m_this, iElementNumber);
	}

	/**
	 * ** This is a special purpose method and not for general consumption **
	 * @param iElementNumber
	 * @return 32 bit signed integer
	 * @throws XFlaimException
	 */
	public int getUINT(
		int		iElementNumber) throws XFlaimException
	{
		return _getUINT( m_this, iElementNumber);
	}

	/**
	 * Method to get the value of the element specified (iElementNumber) of the
	 * target key as a long.  An long is a 64 bit signed value.
	 * @param iElementNumber
	 * @return 64 bit signed integer
	 * @throws XFlaimException
	 */
	public long getLong(
		int		iElementNumber) throws XFlaimException
	{
		return _getLong( m_this, iElementNumber);
	}
	
	/**
	 * Method to get the value of the element specified (iElementNumber) of the
	 * target key as a String.
	 * @param iElementNumber
	 * @return String
	 * @throws XFlaimException
	 */
	public String getString(
		int		iElementNumber) throws XFlaimException
	{
		return _getString( m_this, iElementNumber);
	}

	/**
	 * Method to get the value of the element specified (iElementNumber) of the
	 * target key as binary data.
	 * @param iElementNumber
	 * @return Returns a byte array containing the value of the specified
	 * element
	 * @throws XFlaimException
	 */
	public byte[] getBinary(
		int		iElementNumber) throws XFlaimException
	{
		return _getBinary( m_this, iElementNumber);
	}

	/**
	 * Method to generate a buffer that holds the target key as stored in 
	 * the index.
	 * @param jDb
	 * @param iIndexNum
	 * @param bOutputIds
	 * @return byte[] key buffer
	 * @throws XFlaimException
	 */
	public byte[] outputKey(
		Db		jDb,
		int			iIndexNum,
		boolean		bOutputIds) throws XFlaimException
	{
		return _outputKey( m_this, jDb.m_this, iIndexNum, bOutputIds);
	}

	/**
	 * Method to generate a buffer that holds only the data of the target key.
	 * @param jDb
	 * @param iIndexNum
	 * @return byte[]
	 * @throws XFlaimException
	 */
	public byte[] outputData(
		Db			jDb,
		int		iIndexNum) throws XFlaimException
	{
		return _outputData( m_this, jDb.m_this, iIndexNum);
	}

	/**
	 * Method to populate a DataVector object from an index key.
	 * @param jDb
	 * @param iIndexNum
	 * @param Key
	 * @param iKeyLen
	 * @throws XFlaimException
	 */
	public void inputKey(
		Db				jDb,
		int			iIndexNum,
		byte[]		Key,
		int			iKeyLen) throws XFlaimException
	{
		_inputKey( m_this, jDb.m_this, iIndexNum, Key, iKeyLen);
	}

	/**
	 * Method to populate a portion of a DataVector object from the data part of 
	 * an index key.
	 * @param jDb
	 * @param iIndexNum
	 * @throws XFlaimException
	 */
	public void inputData(
		Db				jDb,
		int			iIndexNum,
		byte[]		Data,
		int			iDataLen) throws XFlaimException
	{
		_inputData( m_this, jDb.m_this, iIndexNum, Data, iDataLen);
	}
	
	/**
	 * Method to reset the contents of the DataVector object.
	 */
	public void reset()
	{
		_reset( m_this);
	}

	/**
	 * Desc:
	 */
	private native void _release( 
		long 		lThis);
	
	/**
	 * Desc:
	 */
	private native void _setDocumentId(
		long		lThis,
		long		lDocId);

	/**
	 * Desc:
	 */
	private native void _setID(
		long		lThis,
		int		iElementNumber,
		long		lID);
		
	/**
	 * Desc:
	 */
	private native void _setNameId(
		long		lThis,
		int		iElementNumber,
		int		iNameId,
		boolean	bIsAttr,
		boolean	bIsData);
		
	/**
	 * Desc:
	 */
	private native void _setINT(
		long		lThis,
		int		iElementNumber,
		int		iNum);
		
	/**
	 * Desc:
	 */
	private native void _setUINT(
		long		lThis,
		int		iElementNumber,
		int		iUNum);
		
	/**
	 * Desc:
	 */
	private native void _setLong(
		long		lThis,
		int		iElementNumber,
		long		lNum);
		
	/**
	 * Desc:
	 */
	private native void _setString(
		long		lThis,
		int		iElementNumber,
		String	sValue);
	
	/**
	 * Desc:
	 */
	private native void _setBinary(
		long		lThis,
		int		iElementNumber,
		byte[]	Value);
	
	/**
	 * Desc:
	 */
	private native void _setRightTruncated(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native void _setLeftTruncated(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native void _clearRightTruncated(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native void _clearLeftTruncated(
		long		lThis,
		int		iElementNumber);
	
	/**
	 * Desc:
	 */
	private native long _getDocumentID(
		long		lThis);
	
	/**
	 * Desc:
	 */
	private native long _getID(
		long		lThis,
		int		iElementNumber);
		
	/**
	 * Desc:
	 */
	private native int _getNameId(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native boolean _isAttr(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native boolean _isDataComponent(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native boolean _isKeyComponent(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native int _getDataLength(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native int _getDataType(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native int _getINT(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native int _getUINT(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native long _getLong(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native String _getString(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native byte[] _getBinary(
		long		lThis,
		int		iElementNumber);

	/**
	 * Desc:
	 */
	private native byte[] _outputKey(
		long		lThis,
		long		ljDbRef,
		int		iIndexNum,
		boolean	bOutputIds);

	/**
	 * Desc:
	 */
	private native byte[] _outputData(
		long		lThis,
		long		ljDbRef,
		int		iIndexNum);

	/**
	 * Desc:
	 */
	private native void _inputKey(
		long		lThis,
		long		ljDbRef,
		int		iIndexNum,
		byte[]	Key,
		int		iKeyLen);

	/**
	 * Desc:
	 */
	private native void _inputData(
		long		lThis,
		long		ljDbRef,
		int		iIndexNum,
		byte[]	Data,
		int		iDataLen);

	/**
	 * Desc:
	 */
	private native void _reset(
		long		lThis);
}
