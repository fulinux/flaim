//------------------------------------------------------------------------------
// Desc:	RCODEs
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
// $Id: RCODE.java 3110 2006-01-19 13:09:08 -0700 (Thu, 19 Jan 2006; dsanders $
//------------------------------------------------------------------------------

package xflaim;

/**
 * Provides enums for some of the error codes that XFlaim might return in an
 * {@link xflaim.XFlaimException XFlaimException}.
 */

public final class RCODE
{
	public static final int NE_XFLM_OK										= 0;
	
	public static final int NE_XFLM_FIRST_COMMON_ERROR					= 0x81050000;			// NOTE: This is not an error code - do not document it
	public static final int NE_XFLM_NOT_IMPLEMENTED						= 0x81050001;			// NE_NOT_IMPLEMENTED - Attempt was made to use a feature that is not implemented.
	public static final int NE_XFLM_MEM										= 0x81050002;			// NE_INSUFFICIENT_MEMORY - Attempt to allocate memory failed.
	public static final int NE_XFLM_INVALID_PARM							= 0x81050005;			// NE_INVALID_PARAMETER - Invalid parameter passed into a function.
	public static final int NE_XFLM_TIMEOUT								= 0x81050009;			// NE_WAIT_TIMEOUT - Database operation timed out (usually a query operation;.
	public static final int NE_XFLM_NOT_FOUND								= 0x8105000A;			// NE_OBJECT_NOT_FOUND - An object was not found.
	public static final int NE_XFLM_EXISTS									= 0x8105000C;			// NE_OBJECT_ALREADY_EXISTS - Object already exists.
	public static final int NE_XFLM_USER_ABORT							= 0x81050010;			// NE_CALLBACK_CANCELLED - User or application aborted (canceled; the operation
	public static final int NE_XFLM_FAILURE								= 0x81050011;			// NE_RECOVERABLE_FAILURE - Internal failure.
	public static final int NE_XFLM_LAST_COMMON_ERROR					= 0x81050012;			// NOTE: This is not an error code - do not document.
	
	public static final int NE_XFLM_FIRST_GENERAL_ERROR				= 0x81050100;			// NOTE: This is not an error code - do not document
	public static final int NE_XFLM_BOF_HIT								= 0x81050101;			// Beginning of results encountered.  This error is may be returned when reading query results in reverse order (from last to first;.
	public static final int NE_XFLM_EOF_HIT								= 0x81050102;			// End of results encountered.  This error may be returned when reading query results in forward order (first to last;.
	public static final int NE_XFLM_END										= 0x81050103;			// End of roll-forward log packets encountered.  NOTE: This error code should never be returned to an application.
	public static final int NE_XFLM_BAD_PREFIX							= 0x81050104;			// Invalid XLM namespace prefix specified.  Either a prefix name or number that was specified was not defined.
	public static final int NE_XFLM_ATTRIBUTE_PURGED					= 0x81050105;			// XML attribute cannot be used - it is being deleted from the database.
	public static final int NE_XFLM_BAD_COLLECTION						= 0x81050106;			// Invalid collection number specified.  Collection is not defined.
	public static final int NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT		= 0x81050107;			// Request to lock the database timed out.
	public static final int NE_XFLM_ILLEGAL_DATA_COMPONENT			= 0x81050108;			// Cannot use ELM_ROOT_TAG as a data component in an index.
	public static final int NE_XFLM_BAD_DATA_TYPE						= 0x81050109;			// Attempt to set/get data on an XML element or attribute using a data type that is incompatible with the data type specified in the dictionary.
	public static final int NE_XFLM_MUST_INDEX_ON_PRESENCE			= 0x8105010A;			// When using ELM_ROOT_TAG in an index component, must specify PRESENCE indexing only.
	public static final int NE_XFLM_BAD_IX									= 0x8105010B;			// Invalid index number specified.  Index is not defined.
	public static final int NE_XFLM_BACKUP_ACTIVE						= 0x8105010C;			// Operation could not be performed because a backup is currently in progress.
	public static final int NE_XFLM_SERIAL_NUM_MISMATCH				= 0x8105010D;			// Serial number on backup file does not match the serial number that is expected.
	public static final int NE_XFLM_BAD_RFL_DB_SERIAL_NUM				= 0x8105010E;			// Bad database serial number in roll-forward log file header.
	public static final int NE_XFLM_BTREE_ERROR							= 0x8105010F;			// A B-Tree in the database is bad.
	public static final int NE_XFLM_BTREE_FULL							= 0x81050110;			// A B-tree in the database is full, or a b-tree being used for a temporary result set is full.
	public static final int NE_XFLM_BAD_RFL_FILE_NUMBER				= 0x81050111;			// Bad roll-forward log file number in roll-forward log file header.
	public static final int NE_XFLM_CANNOT_DEL_ELEMENT					= 0x81050112;			// Cannot delete an XML element definition in the dictionary because it is in use.
	public static final int NE_XFLM_CANNOT_MOD_DATA_TYPE				= 0x81050113;			// Cannot modify the data type for an XML element or attribute definition in the dictionary.
	public static final int NE_XFLM_CANNOT_INDEX_DATA_TYPE			= 0x81050114;			// Data type of XML element or attribute is not one that can be indexed.
	public static final int NE_XFLM_CONV_BAD_DIGIT						= 0x81050115;			// Non-numeric digit found in text to numeric conversion.
	public static final int NE_XFLM_CONV_DEST_OVERFLOW					= 0x81050116;			// Destination buffer not large enough to hold data.
	public static final int NE_XFLM_CONV_ILLEGAL							= 0x81050117;			// Attempt to convert between data types is an unsupported conversion.
	public static final int NE_XFLM_CONV_NULL_SRC						= 0x81050118;			// Data source cannot be NULL when doing data conversion.
	public static final int NE_XFLM_CONV_NUM_OVERFLOW					= 0x81050119;			// Numeric overflow (> upper bound; converting to numeric type.
	public static final int NE_XFLM_CONV_NUM_UNDERFLOW					= 0x8105011A;			// Numeric underflow (< lower bound; converting to numeric type.
	public static final int NE_XFLM_BAD_ELEMENT_NUM						= 0x8105011B;			// Bad element number specified - element not defined in dictionary.
	public static final int NE_XFLM_BAD_ATTRIBUTE_NUM					= 0x8105011C;			// Bad attribute number specified - attribute not defined in dictionary.
	public static final int NE_XFLM_BAD_ENCDEF_NUM						= 0x8105011D;			// Bad encryption number specified - encryption definition not defined in dictionary.
	public static final int NE_XFLM_DATA_ERROR							= 0x8105011E;			// Encountered data in the database that was corrupted.
	public static final int NE_XFLM_INVALID_FILE_SEQUENCE				= 0x8105011F;			// Incremental backup file number provided during a restore is invalid.
	public static final int NE_XFLM_ILLEGAL_OP							= 0x81050120;			// Attempt to perform an illegal operation.
	public static final int NE_XFLM_DUPLICATE_ELEMENT_NUM				= 0x81050121;			// Element number specified in element definition is already in use.
	public static final int NE_XFLM_ILLEGAL_TRANS_TYPE					= 0x81050122;			// Illegal transaction type specified for transaction begin operation.
	public static final int NE_XFLM_UNSUPPORTED_VERSION				= 0x81050123;			// Version of database found in database header is not supported.
	public static final int NE_XFLM_ILLEGAL_TRANS_OP					= 0x81050124;			// Illegal operation for transaction type.
	public static final int NE_XFLM_INCOMPLETE_LOG						= 0x81050125;			// Incomplete rollback log.
	public static final int NE_XFLM_ILLEGAL_INDEX_DEF					= 0x81050126;			// Index definition document is illegal - does not conform to the expected form of an index definition document.
	public static final int NE_XFLM_ILLEGAL_INDEX_ON					= 0x81050127;			// The "IndexOn" attribute of an index definition has an illegal value.
	public static final int NE_XFLM_ILLEGAL_STATE_CHANGE				= 0x81050128;			// Attempted an illegal state change on an element or attribute definition.
	public static final int NE_XFLM_BAD_RFL_SERIAL_NUM					= 0x81050129;			// Serial number in roll-forward log file header does not match expected serial number.
	public static final int NE_XFLM_NEWER_FLAIM							= 0x8105012A;			// Running old code on a newer version of database.  Newer code must be used.
	public static final int NE_XFLM_CANNOT_MOD_ELEMENT_STATE			= 0x8105012B;			// Attempted to change state of a predefined element definition.
	public static final int NE_XFLM_CANNOT_MOD_ATTRIBUTE_STATE		= 0x8105012C;			// Attempted to change state of a predefined attribute definition.
	public static final int NE_XFLM_NO_MORE_ELEMENT_NUMS				= 0x8105012D;			// The highest element number has already been used, cannot create more element definitions.
	public static final int NE_XFLM_NO_TRANS_ACTIVE						= 0x8105012E;			// Operation must be performed inside a database transaction.
	public static final int NE_XFLM_NOT_UNIQUE							= 0x8105012F;			// Attempt was made to insert a key into a b-tree that was already in the b-tree.
	public static final int NE_XFLM_NOT_FLAIM								= 0x81050130;			// The file specified is not a FLAIM database.
	public static final int NE_XFLM_OLD_VIEW								= 0x81050131;			// Unable to maintain read transaction's view of the database.
	public static final int NE_XFLM_SHARED_LOCK							= 0x81050132;			// Attempted to perform an operation on the database that requires exclusive access, but cannot because there is a shared lock.
	public static final int NE_XFLM_SYNTAX									= 0x81050133;			// Syntax error while parsing XML or query.
	public static final int NE_XFLM_TRANS_ACTIVE							= 0x81050134;			// Operation cannot be performed while a transaction is active.
	public static final int NE_XFLM_RFL_TRANS_GAP						= 0x81050135;			// A gap was found in the transaction sequence in the roll-forward log.
	public static final int NE_XFLM_BAD_COLLATED_KEY					= 0x81050136;			// Something in collated key is bad.
	public static final int NE_XFLM_UNSUPPORTED_FEATURE				= 0x81050137;			// Attempting to use a feature for which full support has been disabled.
	public static final int NE_XFLM_MUST_DELETE_INDEXES				= 0x81050138;			// Attempting to delete a collection that has indexes defined for it.  Associated indexes must be deleted before the collection can be deleted.
	public static final int NE_XFLM_RFL_INCOMPLETE						= 0x81050139;			// Roll-forward log file is incomplete.
	public static final int NE_XFLM_CANNOT_RESTORE_RFL_FILES			= 0x8105013A;			// Cannot restore roll-forward log files - not using multiple roll-forward log files.
	public static final int NE_XFLM_INCONSISTENT_BACKUP				= 0x8105013B;			// A problem (corruption, etc.; was detected in a backup set.
	public static final int NE_XFLM_BLOCK_CRC								= 0x8105013C;			// CRC for database block was invalid.  May indicate problems in reading from or writing to disk.
	public static final int NE_XFLM_ABORT_TRANS							= 0x8105013D;			// Attempted operation after a critical error - transaction should be aborted.
	public static final int NE_XFLM_NOT_RFL								= 0x8105013E;			// File was not a roll-forward log file as expected.
	public static final int NE_XFLM_BAD_RFL_PACKET						= 0x8105013F;			// Roll-forward log file packet was bad.
	public static final int NE_XFLM_DATA_PATH_MISMATCH					= 0x81050140;			// Bad data path specified to open database.  Does not match data path specified for prior opens of the database.
	public static final int NE_XFLM_STREAM_EXISTS						= 0x81050141;			// Attempt to create stream, but the file(s; already exists.
	public static final int NE_XFLM_FILE_EXISTS							= 0x81050142;			// Attempt to create a database, but the file already exists.
	public static final int NE_XFLM_COULD_NOT_CREATE_SEMAPHORE		= 0x81050143;			// Could not create a semaphore.
	public static final int NE_XFLM_MUST_CLOSE_DATABASE				= 0x81050144;			// Database must be closed due to a critical error.
	public static final int NE_XFLM_INVALID_ENCKEY_CRC					= 0x81050145;			// Encryption key CRC could not be verified.
	public static final int NE_XFLM_BAD_UTF8								= 0x81050146;			// An invalid byte sequence was found in a UTF-8 string
	public static final int NE_XFLM_COULD_NOT_CREATE_MUTEX			= 0x81050147;			// Could not create a mutex.
	public static final int NE_XFLM_ERROR_WAITING_ON_SEMPAHORE		= 0x81050148;			// Error occurred while waiting on a sempahore.
	public static final int NE_XFLM_BAD_PLATFORM_FORMAT				= 0x81050149;			// Cannot support platform format.  NOTE: No need to document this one, it is strictly internal.
	public static final int NE_XFLM_HDR_CRC								= 0x8105014A;			// Database header has a bad CRC.
	public static final int NE_XFLM_NO_NAME_TABLE						= 0x8105014B;			// No name table was set up for the database.
	public static final int NE_XFLM_MULTIPLE_MATCHES					= 0x8105014C;			// Multiple entries match the name in the name table.  Need to pass a namespace to disambiguate.
	public static final int NE_XFLM_UNALLOWED_UPGRADE					= 0x8105014D;			// Cannot upgrade database from one version to another.
	public static final int NE_XFLM_BTREE_BAD_STATE						= 0x8105014E;			// Btree function called before proper setup steps taken.
	public static final int NE_XFLM_DUPLICATE_ATTRIBUTE_NUM			= 0x8105014F;			// Attribute number specified in attribute definition is already in use.
	public static final int NE_XFLM_DUPLICATE_INDEX_NUM				= 0x81050150;			// Index number specified in index definition is already in use.
	public static final int NE_XFLM_DUPLICATE_COLLECTION_NUM			= 0x81050151;			// Collection number specified in collection definition is already in use.
	public static final int NE_XFLM_DUPLICATE_ELEMENT_NAME			= 0x81050152;			// Element name+namespace specified in element definition is already in use.
	public static final int NE_XFLM_DUPLICATE_ATTRIBUTE_NAME			= 0x81050153;			// Attribute name+namespace specified in attribute definition is already in use.
	public static final int NE_XFLM_DUPLICATE_INDEX_NAME				= 0x81050154;			// Index name specified in index definition is already in use.
	public static final int NE_XFLM_DUPLICATE_COLLECTION_NAME		= 0x81050155;			// Collection name specified in collection definition is already in use.
	public static final int NE_XFLM_ELEMENT_PURGED						= 0x81050156;			// XML element cannot be used - it is deleted from the database.
	public static final int NE_XFLM_TOO_MANY_OPEN_DATABASES			= 0x81050157;			// Too many open databases, cannot open another one.
	public static final int NE_XFLM_DATABASE_OPEN						= 0x81050158;			// Operation cannot be performed because the database is currently open.
	public static final int NE_XFLM_CACHE_ERROR							= 0x81050159;			// Cached database block has been compromised while in cache.
	public static final int NE_XFLM_BTREE_KEY_SIZE						= 0x8105015A;			// Key too large to insert/lookup in a b-tree.
	public static final int NE_XFLM_DB_FULL								= 0x8105015B;			// Database is full, cannot create more blocks.
	public static final int NE_XFLM_QUERY_SYNTAX							= 0x8105015C;			// Query expression had improper syntax.
	public static final int NE_XFLM_COULD_NOT_START_THREAD			= 0x8105015D;			// Error occurred while attempting to start a thread.
	public static final int NE_XFLM_INDEX_OFFLINE						= 0x8105015E;			// Index is offline, cannot be used in a query.
	public static final int NE_XFLM_RFL_DISK_FULL						= 0x8105015F;			// Disk which contains roll-forward log is full.
	public static final int NE_XFLM_MUST_WAIT_CHECKPOINT				= 0x81050160;			// Must wait for a checkpoint before starting transaction - due to disk problems - usually in disk containing roll-forward log files.
	public static final int NE_XFLM_MISSING_ENC_ALGORITHM				= 0x81050161;			// Encryption definition is missing an encryption algorithm.
	public static final int NE_XFLM_INVALID_ENC_ALGORITHM				= 0x81050162;			// Invalid encryption algorithm specified in encryption definition.
	public static final int NE_XFLM_INVALID_ENC_KEY_SIZE				= 0x81050163;			// Invalid key size specified in encryption definition.
	public static final int NE_XFLM_ILLEGAL_DATA_TYPE					= 0x81050164;			// Data type specified for XML element or attribute definition is illegal.
	public static final int NE_XFLM_ILLEGAL_STATE						= 0x81050165;			// State specified for index definition or XML element or attribute definition is illegal.
	public static final int NE_XFLM_ILLEGAL_ELEMENT_NAME				= 0x81050166;			// XML element name specified in element definition is illegal.
	public static final int NE_XFLM_ILLEGAL_ATTRIBUTE_NAME			= 0x81050167;			// XML attribute name specified in attribute definition is illegal.
	public static final int NE_XFLM_ILLEGAL_COLLECTION_NAME			= 0x81050168;			// Collection name specified in collection definition is illegal.
	public static final int NE_XFLM_ILLEGAL_INDEX_NAME					= 0x81050169;			// Index name specified is illegal
	public static final int NE_XFLM_ILLEGAL_ELEMENT_NUMBER			= 0x8105016A;			// Element number specified in element definition or index definition is illegal.
	public static final int NE_XFLM_ILLEGAL_ATTRIBUTE_NUMBER			= 0x8105016B;			// Attribute number specified in attribute definition or index definition is illegal.
	public static final int NE_XFLM_ILLEGAL_COLLECTION_NUMBER		= 0x8105016C;			// Collection number specified in collection definition or index definition is illegal.
	public static final int NE_XFLM_ILLEGAL_INDEX_NUMBER				= 0x8105016D;			// Index number specified in index definition is illegal.
	public static final int NE_XFLM_ILLEGAL_ENCDEF_NUMBER				= 0x8105016E;			// Encryption definition number specified in encryption definition is illegal.
	public static final int NE_XFLM_COLLECTION_NAME_MISMATCH			= 0x8105016F;			// Collection name and number specified in index definition do not correspond to each other.
	public static final int NE_XFLM_ELEMENT_NAME_MISMATCH				= 0x81050170;			// Element name+namespace and number specified in index definition do not correspond to each other.
	public static final int NE_XFLM_ATTRIBUTE_NAME_MISMATCH			= 0x81050171;			// Attribute name+namespace and number specified in index definition do not correspond to each other.
	public static final int NE_XFLM_INVALID_COMPARE_RULE				= 0x81050172;			// Invalid comparison rule specified in index definition.
	public static final int NE_XFLM_DUPLICATE_KEY_COMPONENT			= 0x81050173;			// Duplicate key component number specified in index definition.
	public static final int NE_XFLM_DUPLICATE_DATA_COMPONENT			= 0x81050174;			// Duplicate data component number specified in index definition.
	public static final int NE_XFLM_MISSING_KEY_COMPONENT				= 0x81050175;			// Index definition is missing a key component.
	public static final int NE_XFLM_MISSING_DATA_COMPONENT			= 0x81050176;			// Index definition is missing a data component.
	public static final int NE_XFLM_INVALID_INDEX_OPTION				= 0x81050177;			// Invalid index option specified on index definition.
	public static final int NE_XFLM_NO_MORE_ATTRIBUTE_NUMS			= 0x81050178;			// The highest attribute number has already been used, cannot create more.
	public static final int NE_XFLM_MISSING_ELEMENT_NAME				= 0x81050179;			// Missing element name in XML element definition.
	public static final int NE_XFLM_MISSING_ATTRIBUTE_NAME			= 0x8105017A;			// Missing attribute name in XML attribute definition.
	public static final int NE_XFLM_MISSING_ELEMENT_NUMBER			= 0x8105017B;			// Missing element number in XML element definition.
	public static final int NE_XFLM_MISSING_ATTRIBUTE_NUMBER			= 0x8105017C;			// Missing attribute number from XML attribute definition.
	public static final int NE_XFLM_MISSING_INDEX_NAME					= 0x8105017D;			// Missing index name in index definition.
	public static final int NE_XFLM_MISSING_INDEX_NUMBER				= 0x8105017E;			// Missing index number in index definition.
	public static final int NE_XFLM_MISSING_COLLECTION_NAME			= 0x8105017F;			// Missing collection name in collection definition.
	public static final int NE_XFLM_MISSING_COLLECTION_NUMBER		= 0x81050180;			// Missing collection number in collection definition.
	public static final int NE_XFLM_BAD_SEN								= 0x81050181;			// Invalid simple encoded number.
	public static final int NE_XFLM_MISSING_ENCDEF_NAME				= 0x81050182;			// Missing encryption definition name in encryption definition.
	public static final int NE_XFLM_MISSING_ENCDEF_NUMBER				= 0x81050183;			// Missing encryption definition number in encryption definition.
	public static final int NE_XFLM_NO_MORE_INDEX_NUMS					= 0x81050184;			// The highest index number has already been used, cannot create more.
	public static final int NE_XFLM_NO_MORE_COLLECTION_NUMS			= 0x81050185;			// The highest collection number has already been used, cannot create more.
	public static final int NE_XFLM_CANNOT_DEL_ATTRIBUTE				= 0x81050186;			// Cannot delete an XML attribute definition because it is in use.
	public static final int NE_XFLM_TOO_MANY_PENDING_NODES			= 0x81050187;			// Too many documents in the pending document list.
	public static final int NE_XFLM_UNSUPPORTED_INTERFACE				= 0x81050188;			// Requested COM interface is not supported.
	public static final int NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG			= 0x81050189;			// ELM_ROOT_TAG, if used, must be the sole root component of an index definition.
	public static final int NE_XFLM_DUP_SIBLING_IX_COMPONENTS		= 0x8105018A;			// Sibling components in an index definition cannot have the same XML element or attribute number.
	public static final int NE_XFLM_RFL_FILE_NOT_FOUND					= 0x8105018B;			// Could not open a roll-forward log file - was not found in the roll-forward log directory.
	public static final int NE_XFLM_BAD_RCODE_TABLE						= 0x8105018C;			// The error code tables are incorrect.  NOTE: This is an internal error that does not need to be documented.
	public static final int NE_XFLM_ILLEGAL_KEY_COMPONENT_NUM		= 0x8105018D;			// Key component of zero in index definition is not allowed.
	public static final int NE_XFLM_ILLEGAL_DATA_COMPONENT_NUM		= 0x8105018E;			// Data component of zero in index definition is not allowed.
	public static final int NE_XFLM_CLASS_NOT_AVAILABLE				= 0x8105018F;			// Requested COM class is not available.
	public static final int NE_XFLM_BUFFER_OVERFLOW						= 0x81050190;			// Buffer overflow.
	public static final int NE_XFLM_ILLEGAL_PREFIX_NUMBER				= 0x81050191;			// Prefix number specified in prefix definition is illegal.
	public static final int NE_XFLM_MISSING_PREFIX_NAME				= 0x81050192;			// Missing prefix name in prefix definition.
	public static final int NE_XFLM_MISSING_PREFIX_NUMBER				= 0x81050193;			// Missing prefix number in prefix definition.
	public static final int NE_XFLM_UNDEFINED_ELEMENT_NAME			= 0x81050194;			// XML element name+namespace that was specified in index definition or XML document is not defined in dictionary.
	public static final int NE_XFLM_UNDEFINED_ATTRIBUTE_NAME			= 0x81050195;			// XML attribute name+namespace that was specified in index definition or XML document is not defined in dictionary.
	public static final int NE_XFLM_DUPLICATE_PREFIX_NAME				= 0x81050196;			// Prefix name specified in prefix definition is already in use.
	public static final int NE_XFLM_KEY_OVERFLOW							= 0x81050197;			// Generated index key too large.
	public static final int NE_XFLM_UNESCAPED_METACHAR					= 0x81050198;			// Unescaped metacharacter in regular expression.
	public static final int NE_XFLM_ILLEGAL_QUANTIFIER					= 0x81050199;			// Illegal quantifier in regular expression.
	public static final int NE_XFLM_UNEXPECTED_END_OF_EXPR			= 0x8105019A;			// Unexpected end of regular expression.
	public static final int NE_XFLM_ILLEGAL_MIN_COUNT					= 0x8105019B;			// Illegal minimum count in regular expression quantifier.
	public static final int NE_XFLM_ILLEGAL_MAX_COUNT					= 0x8105019C;			// Illegal maximum count in regular expression quantifier.
	public static final int NE_XFLM_EMPTY_BRANCH_IN_EXPR				= 0x8105019D;			// Illegal empty branch in a regular expression.
	public static final int NE_XFLM_ILLEGAL_RPAREN_IN_EXPR			= 0x8105019E;			// Illegal right paren in a regular expression.
	public static final int NE_XFLM_ILLEGAL_CLASS_SUBTRACTION		= 0x8105019F;			// Illegal class subtraction in regular expression.
	public static final int NE_XFLM_ILLEGAL_CHAR_RANGE_IN_EXPR		= 0x810501A0;			// Illegal character range in regular expression.
	public static final int NE_XFLM_BAD_BASE64_ENCODING				= 0x810501A1;			// Illegal character(s; found in a base64 stream.
	public static final int NE_XFLM_NAMESPACE_NOT_ALLOWED				= 0x810501A2;			// Cannot define a namespace for XML attributes whose name begins with "xmlns:" or that is equal to "xmlns"
	public static final int NE_XFLM_INVALID_NAMESPACE_DECL			= 0x810501A3;			// Name for namespace declaration attribute must be "xmlns" or begin with "xmlns:"
	public static final int NE_XFLM_ILLEGAL_NAMESPACE_DECL_DATATYPE= 0x810501A4;			// Data type for XML attributes that are namespace declarations must be text.
	public static final int NE_XFLM_UNEXPECTED_END_OF_INPUT			= 0x810501A5;		   // Encountered unexpected end of input when parsing XPATH expression.
	public static final int NE_XFLM_NO_MORE_PREFIX_NUMS				= 0x810501A6;			// The highest prefix number has already been used, cannot create more.
	public static final int NE_XFLM_NO_MORE_ENCDEF_NUMS				= 0x810501A7;			// The highest encryption definition number has already been used, cannot create more.
	public static final int NE_XFLM_COLLECTION_OFFLINE					= 0x810501A8;			// Collection is encrypted, cannot be accessed while in operating in limited mode.
	public static final int NE_XFLM_INVALID_XML							= 0x810501A9;			// Invalid XML encountered while parsing document.
	public static final int NE_XFLM_READ_ONLY								= 0x810501AA;			// Item is read-only and cannot be updated.
	public static final int NE_XFLM_DELETE_NOT_ALLOWED					= 0x810501AB;			// Item cannot be deleted.
	public static final int NE_XFLM_RESET_NEEDED							= 0x810501AC;			// Used during check operations to indicate we need to reset the view.  NOTE: This is an internal error code and should not be documented.
	public static final int NE_XFLM_ILLEGAL_REQUIRED_VALUE			= 0x810501AD;			// An illegal value was specified for the "Required" attribute in an index definition.
	public static final int NE_XFLM_ILLEGAL_INDEX_COMPONENT			= 0x810501AE;			// A leaf index component in an index definition was not marked as a data component or key component.
	public static final int NE_XFLM_ILLEGAL_UNIQUE_SUB_ELEMENT_VALUE	= 0x810501AF;		// Illegal value for the "UniqueSubElements" attribute in an element definition.
	public static final int NE_XFLM_DATA_TYPE_MUST_BE_NO_DATA		= 0x810501B0;			// Data type for an element definition with UniqueSubElements="yes" must be nodata.
	public static final int NE_XFLM_ILLEGAL_FLAG							= 0x810501B1;			// Illegal flag passed to getChildElement method.  Must be zero for elements that can have non-unique child elements.
	public static final int NE_XFLM_CANNOT_SET_REQUIRED				= 0x810501B2;			// Cannot set the "Required" attribute on a non-key index component in index definition.
	public static final int NE_XFLM_CANNOT_SET_LIMIT					= 0x810501B3;			// Cannot set the "Limit" attribute on a non-key index component in index definition.
	public static final int NE_XFLM_CANNOT_SET_INDEX_ON				= 0x810501B4;			// Cannot set the "IndexOn" attribute on a non-key index component in index definition.
	public static final int NE_XFLM_CANNOT_SET_COMPARE_RULES			= 0x810501B5;			// Cannot set the "CompareRules" on a non-key index component in index definition.
	public static final int NE_XFLM_INPUT_PENDING						= 0x810501B6;			// Attempt to set a value while an input stream is still open.
	public static final int NE_XFLM_INVALID_NODE_TYPE					= 0x810501B7;			// Bad node type
	public static final int NE_XFLM_INVALID_CHILD_ELM_NODE_ID		= 0x810501B8;			// Attempt to insert a unique child element that has a lower node ID than the parent element
	public static final int NE_XFLM_LAST_GENERAL_ERROR					= 0x810501B9;			// NOTE: This is not an error code - do not document

	/****************************************************************************
	Desc:		DOM Errors
	****************************************************************************/

	public static final int NE_XFLM_FIRST_DOM_ERROR						= 0x81051100;			// NOTE: This is not an error code - do not document
	public static final int NE_XFLM_DOM_HIERARCHY_REQUEST_ERR		= 0x81051101;			// Attempt to insert a DOM node somewhere it doesn't belong.
	public static final int NE_XFLM_DOM_WRONG_DOCUMENT_ERR			= 0x81051102;			// A DOM node is being used in a different document than the one that created it.
	public static final int NE_XFLM_DOM_DATA_ERROR						= 0x81051103;			// Links between DOM nodes in a document are corrupt.
	public static final int NE_XFLM_DOM_NODE_NOT_FOUND					= 0x81051104;			// The requested DOM node does not exist.
	public static final int NE_XFLM_DOM_INVALID_CHILD_TYPE			= 0x81051105;			// Attempting to insert a child DOM node whose type cannot be inserted as a child node.
	public static final int NE_XFLM_DOM_NODE_DELETED					= 0x81051106;			// DOM node being accessed has been deleted.
	public static final int NE_XFLM_DOM_DUPLICATE_ELEMENT				= 0x81051107;			// Node already has a child element with the given name id - this node's child nodes must all be unique.
	public static final int NE_XFLM_LAST_DOM_ERROR						= 0x81051108;			// NOTE: This is not an error code - do not document

	/****************************************************************************
	Desc:		I/O Errors
	****************************************************************************/

	public static final int NE_XFLM_FIRST_IO_ERROR						= 0x81052100;			// NOTE: This is not an error code - do not document
	public static final int NE_XFLM_IO_ACCESS_DENIED					= 0x81052101;			// Access to file is denied. Caller is not allowed access to a file.
	public static final int NE_XFLM_IO_BAD_FILE_HANDLE					= 0x81052102;			// Bad file handle or file descriptor.
	public static final int NE_XFLM_IO_COPY_ERR							= 0x81052103;			// Error occurred while copying a file.
	public static final int NE_XFLM_IO_DISK_FULL							= 0x81052104;			// Disk full.
	public static final int NE_XFLM_IO_END_OF_FILE						= 0x81052105;			// End of file reached while reading from the file.
	public static final int NE_XFLM_IO_OPEN_ERR							= 0x81052106;			// Error while opening the file.
	public static final int NE_XFLM_IO_SEEK_ERR							= 0x81052107;			// Error occurred while positioning (seeking; within a file.
	public static final int NE_XFLM_IO_DIRECTORY_ERR					= 0x81052108;			// Error occurred while accessing or deleting a directory.
	public static final int NE_XFLM_IO_PATH_NOT_FOUND					= 0x81052109;			// File not found.
	public static final int NE_XFLM_IO_TOO_MANY_OPEN_FILES			= 0x8105210A;			// Too many files open.
	public static final int NE_XFLM_IO_PATH_TOO_LONG					= 0x8105210B;			// File name too long.
	public static final int NE_XFLM_IO_NO_MORE_FILES					= 0x8105210C;			// No more files in directory.
	public static final int NE_XFLM_IO_DELETING_FILE					= 0x8105210D;			// Error occurred while deleting a file.
	public static final int NE_XFLM_IO_FILE_LOCK_ERR					= 0x8105210E;			// Error attempting to acquire a byte-range lock on a file.
	public static final int NE_XFLM_IO_FILE_UNLOCK_ERR					= 0x8105210F;			// Error attempting to release a byte-range lock on a file.
	public static final int NE_XFLM_IO_PATH_CREATE_FAILURE			= 0x81052110;			// Error occurred while attempting to create a directory or sub-directory.
	public static final int NE_XFLM_IO_RENAME_FAILURE					= 0x81052111;			// Error occurred while renaming a file.
	public static final int NE_XFLM_IO_INVALID_PASSWORD				= 0x81052112;			// Invalid file password.
	public static final int NE_XFLM_SETTING_UP_FOR_READ				= 0x81052113;			// Error occurred while setting up to perform a file read operation.
	public static final int NE_XFLM_SETTING_UP_FOR_WRITE				= 0x81052114;			// Error occurred while setting up to perform a file write operation.
	public static final int NE_XFLM_IO_CANNOT_REDUCE_PATH				= 0x81052115;			// Cannot reduce file name into more components.
	public static final int NE_XFLM_INITIALIZING_IO_SYSTEM			= 0x81052116;			// Error occurred while setting up to access the file system.
	public static final int NE_XFLM_FLUSHING_FILE						= 0x81052117;			// Error occurred while flushing file data buffers to disk.
	public static final int NE_XFLM_IO_INVALID_FILENAME				= 0x81052118;			// Invalid file name.
	public static final int NE_XFLM_IO_CONNECT_ERROR					= 0x81052119;			// Error connecting to a remote network resource.
	public static final int NE_XFLM_OPENING_FILE							= 0x8105211A;			// Unexpected error occurred while opening a file.
	public static final int NE_XFLM_DIRECT_OPENING_FILE				= 0x8105211B;			// Unexpected error occurred while opening a file in direct access mode.
	public static final int NE_XFLM_CREATING_FILE						= 0x8105211C;			// Unexpected error occurred while creating a file.
	public static final int NE_XFLM_DIRECT_CREATING_FILE				= 0x8105211D;			// Unexpected error occurred while creating a file in direct access mode.
	public static final int NE_XFLM_READING_FILE							= 0x8105211E;			// Unexpected error occurred while reading a file.
	public static final int NE_XFLM_DIRECT_READING_FILE				= 0x8105211F;			// Unexpected error occurred while reading a file in direct access mode.
	public static final int NE_XFLM_WRITING_FILE							= 0x81052120;			// Unexpected error occurred while writing to a file.
	public static final int NE_XFLM_DIRECT_WRITING_FILE				= 0x81052121;			// Unexpected error occurred while writing a file in direct access mode.
	public static final int NE_XFLM_POSITIONING_IN_FILE				= 0x81052122;			// Unexpected error occurred while positioning within a file.
	public static final int NE_XFLM_GETTING_FILE_SIZE					= 0x81052123;			// Unexpected error occurred while getting a file's size.
	public static final int NE_XFLM_TRUNCATING_FILE						= 0x81052124;			// Unexpected error occurred while truncating a file.
	public static final int NE_XFLM_PARSING_FILE_NAME					= 0x81052125;			// Unexpected error occurred while parsing a file's name.
	public static final int NE_XFLM_CLOSING_FILE							= 0x81052126;			// Unexpected error occurred while closing a file.
	public static final int NE_XFLM_GETTING_FILE_INFO					= 0x81052127;			// Unexpected error occurred while getting information about a file.
	public static final int NE_XFLM_EXPANDING_FILE						= 0x81052128;			// Unexpected error occurred while expanding a file.
	public static final int NE_XFLM_CHECKING_FILE_EXISTENCE			= 0x81052129;			// Unexpected error occurred while checking to see if a file exists.
	public static final int NE_XFLM_RENAMING_FILE						= 0x8105212A;			// Unexpected error occurred while renaming a file.
	public static final int NE_XFLM_SETTING_FILE_INFO					= 0x8105212B;			// Unexpected error occurred while setting a file's information.
	public static final int NE_XFLM_LAST_IO_ERROR						= 0x8105212C;			// NOTE: This is not an error code - do not document

	/****************************************************************************
	Desc:		Network Errors
	****************************************************************************/

	public static final int NE_XFLM_FIRST_NET_ERROR						= 0x81053100;			// NOTE: This is not an error code - do not document
	public static final int NE_XFLM_SVR_NOIP_ADDR						= 0x81053101;			// IP address not found
	public static final int NE_XFLM_SVR_SOCK_FAIL						= 0x81053102;			// IP socket failure
	public static final int NE_XFLM_SVR_CONNECT_FAIL					= 0x81053103;			// TCP/IP connection failure
	public static final int NE_XFLM_SVR_BIND_FAIL						= 0x81053104;			// The TCP/IP services on your system may not be configured or installed.  If this POA is not to run Client/Server, use the /notcpip startup switch or disable TCP/IP through the NWADMIN snapin
	public static final int NE_XFLM_SVR_LISTEN_FAIL						= 0x81053105;			// TCP/IP listen failed
	public static final int NE_XFLM_SVR_ACCEPT_FAIL						= 0x81053106;			// TCP/IP accept failed
	public static final int NE_XFLM_SVR_SELECT_ERR						= 0x81053107;			// TCP/IP select failed
	public static final int NE_XFLM_SVR_SOCKOPT_FAIL					= 0x81053108;			// TCP/IP socket operation failed
	public static final int NE_XFLM_SVR_DISCONNECT						= 0x81053109;			// TCP/IP disconnected
	public static final int NE_XFLM_SVR_READ_FAIL						= 0x8105310A;			// TCP/IP read failed
	public static final int NE_XFLM_SVR_WRT_FAIL							= 0x8105310B;			// TCP/IP write failed
	public static final int NE_XFLM_SVR_READ_TIMEOUT					= 0x8105310C;			// TCP/IP read timeout
	public static final int NE_XFLM_SVR_WRT_TIMEOUT						= 0x8105310D;			// TCP/IP write timeout
	public static final int NE_XFLM_SVR_ALREADY_CLOSED					= 0x8105310E;			// Connection already closed
	public static final int NE_XFLM_LAST_NET_ERROR						= 0x8105310F;			// NOTE: This is not an error code - do not document

	/****************************************************************************
	Desc:	Query Errors
	****************************************************************************/

	public static final int NE_XFLM_FIRST_QUERY_ERROR					= 0x81054100;			// NOTE: This is not an error code - do not document
	public static final int NE_XFLM_Q_UNMATCHED_RPAREN					= 0x81054101;			// Query setup error: Unmatched right paren.
	public static final int NE_XFLM_Q_UNEXPECTED_LPAREN				= 0x81054102;			// Query setup error: Unexpected left paren.
	public static final int NE_XFLM_Q_UNEXPECTED_RPAREN				= 0x81054103;			// Query setup error: Unexpected right paren.
	public static final int NE_XFLM_Q_EXPECTING_OPERAND				= 0x81054104;			// Query setup error: Expecting an operand.
	public static final int NE_XFLM_Q_EXPECTING_OPERATOR				= 0x81054105;			// Query setup error: Expecting an operator.
	public static final int NE_XFLM_Q_UNEXPECTED_COMMA					= 0x81054106;			// Query setup error: Unexpected comma.
	public static final int NE_XFLM_Q_EXPECTING_LPAREN					= 0x81054107;			// Query setup error: Expecting a left paren.
	public static final int NE_XFLM_Q_UNEXPECTED_VALUE					= 0x81054108;			// Query setup error: Unexpected value.
	public static final int NE_XFLM_Q_INVALID_NUM_FUNC_ARGS			= 0x81054109;			// Query setup error: Invalid number of arguments for a function.
	public static final int NE_XFLM_Q_UNEXPECTED_XPATH_COMPONENT	= 0x8105410A;			// Query setup error: Unexpected XPATH componenent.
	public static final int NE_XFLM_Q_ILLEGAL_LBRACKET					= 0x8105410B;			// Query setup error: Illegal left bracket ([;.
	public static final int NE_XFLM_Q_ILLEGAL_RBRACKET					= 0x8105410C;			// Query setup error: Illegal right bracket (];.
	public static final int NE_XFLM_Q_ILLEGAL_OPERAND					= 0x8105410D;			// Query setup error: Operand for some operator is not valid for that operator type.
	public static final int NE_XFLM_Q_ALREADY_OPTIMIZED				= 0x8105410E;			// Operation is illegal, cannot change certain things after query has been optimized.
	public static final int NE_XFLM_Q_MISMATCHED_DB						= 0x8105410F;			// Database handle passed in does not match database associated with query.
	public static final int NE_XFLM_Q_ILLEGAL_OPERATOR					= 0x81054110;			// Illegal operator - cannot pass this operator into the addOperator method.
	public static final int NE_XFLM_Q_ILLEGAL_COMPARE_RULES			= 0x81054111;			// Illegal combination of comparison rules passed to addOperator method.
	public static final int NE_XFLM_Q_INCOMPLETE_QUERY_EXPR			= 0x81054112;			// Query setup error: Query expression is incomplete.
	public static final int NE_XFLM_Q_NOT_POSITIONED					= 0x81054113;			// Query not positioned due to previous error, cannot call getNext, getPrev, or getCurrent
	public static final int NE_XFLM_Q_INVALID_NODE_ID_VALUE			= 0x81054114;			// Query setup error: Invalid type of value constant used for node id value comparison.
	public static final int NE_XFLM_Q_INVALID_META_DATA_TYPE			= 0x81054115;			// Query setup error: Invalid meta data type specified.
	public static final int NE_XFLM_Q_NEW_EXPR_NOT_ALLOWED			= 0x81054116;			// Query setup error: Cannot add an expression to an XPATH component after having added an expression that tests context position.
	public static final int NE_XFLM_Q_INVALID_CONTEXT_POS				= 0x81054117;			// Invalid context position value encountered - must be a positive number.
	public static final int NE_XFLM_Q_INVALID_FUNC_ARG					= 0x81054118;			// Query setup error: Parameter to user-defined functions must be a single XPATH only.
	public static final int NE_XFLM_Q_EXPECTING_RPAREN					= 0x81054119;			// Query setup error: Expecting right paren.
	public static final int NE_XFLM_Q_TOO_LATE_TO_ADD_SORT_KEYS		= 0x8105411A;			// Query setup error: Cannot add sort keys after having called getFirst, getLast, getNext, or getPrev.
	public static final int NE_XFLM_Q_INVALID_SORT_KEY_COMPONENT	= 0x8105411B;			// Query setup error: Invalid sort key component number specified in query.
	public static final int NE_XFLM_Q_DUPLICATE_SORT_KEY_COMPONENT	= 0x8105411C;			// Query setup error: Duplicate sort key component number specified in query.
	public static final int NE_XFLM_Q_MISSING_SORT_KEY_COMPONENT	= 0x8105411D;			// Query setup error: Missing sort key component number in sort keys that were specified for query.
	public static final int NE_XFLM_Q_NO_SORT_KEY_COMPONENTS_SPECIFIED	= 0x8105411E;	// Query setup error: addSortKeys was called, but no sort key components were specified.
	public static final int NE_XFLM_Q_SORT_KEY_CONTEXT_MUST_BE_ELEMENT	= 0x8105411F;	// Query setup error: A sort key context cannot be an XML attribute.
	public static final int NE_XFLM_Q_INVALID_ELEMENT_NUM_IN_SORT_KEYS 	= 0x81054120;	// Query setup error: The XML element number specified for a sort key in a query is invalid - no element definition in the dictionary.
	public static final int NE_XFLM_Q_INVALID_ATTR_NUM_IN_SORT_KEYS = 0x81054121;			// Query setup error: The XML attribute number specified for a sort key in a query is invalid - no attribute definition in the dictionary.
	public static final int NE_XFLM_Q_NON_POSITIONABLE_QUERY			= 0x81054122;			// Attempt is being made to position in a query that is not positionable.
	public static final int NE_XFLM_Q_INVALID_POSITION					= 0x81054123;			// Attempt is being made to position to an invalid position in the result set.
	public static final int NE_XFLM_LAST_QUERY_ERROR					= 0x81054124;			// NOTE: This is not an error code - do not document

	/****************************************************************************
	Desc:	Stream Errors
	****************************************************************************/

	public static final int NE_XFLM_FIRST_STREAM_ERROR					= 0x81056100;			// NOTE: This is not an error code - do not document
	public static final int NE_XFLM_STREAM_DECOMPRESS_ERROR			= 0x81056101;			// Error decompressing data stream.
	public static final int NE_XFLM_STREAM_NOT_COMPRESSED				= 0x81056102;			// Attempting to decompress a data stream that is not compressed.
	public static final int NE_XFLM_STREAM_TOO_MANY_FILES				= 0x81056103;			// Too many files in input stream.
	public static final int NE_XFLM_LAST_STREAM_ERROR					= 0x81056104;			// NOTE: This is not an error code - do not document

	/****************************************************************************
	Desc:	NICI / Encryption Errors
	****************************************************************************/

	public static final int NE_XFLM_FIRST_NICI_ERROR					= 0x81057100;			// NOTE: This is not an error code - do not document
	public static final int NE_XFLM_NICI_CONTEXT							= 0x81057101;			// Error occurred while creating NICI context for encryption/decryption.
	public static final int NE_XFLM_NICI_ATTRIBUTE_VALUE				= 0x81057102;			// Error occurred while accessing an attribute on a NICI encryption key.
	public static final int NE_XFLM_NICI_BAD_ATTRIBUTE					= 0x81057103;			// Value retrieved from an attribute on a NICI encryption key was bad.
	public static final int NE_XFLM_NICI_WRAPKEY_FAILED				= 0x81057104;			// Error occurred while wrapping a NICI encryption key in another NICI encryption key.
	public static final int NE_XFLM_NICI_UNWRAPKEY_FAILED				= 0x81057105;			// Error occurred while unwrapping a NICI encryption key that is wrapped in another NICI encryption key.
	public static final int NE_XFLM_NICI_INVALID_ALGORITHM			= 0x81057106;			// Attempt to use invalid NICI encryption algorithm. 
	public static final int NE_XFLM_NICI_GENKEY_FAILED					= 0x81057107;			// Error occurred while attempting to generate a NICI encryption key.
	public static final int NE_XFLM_NICI_BAD_RANDOM						= 0x81057108;			// Error occurred while generating random data using NICI.
	public static final int NE_XFLM_PBE_ENCRYPT_FAILED					= 0x81057109;			// Error occurred while attempting to wrap a NICI encryption key in a password.
	public static final int NE_XFLM_PBE_DECRYPT_FAILED					= 0x8105710A;			// Error occurred while attempting to unwrap a NICI encryption key that was previously wrapped in a password.
	public static final int NE_XFLM_DIGEST_INIT_FAILED					= 0x8105710B;			// Error occurred while attempting to initialize the NICI digest functionality.
	public static final int NE_XFLM_DIGEST_FAILED						= 0x8105710C;			// Error occurred while attempting to create a NICI digest. 
	public static final int NE_XFLM_INJECT_KEY_FAILED					= 0x8105710D;			// Error occurred while attempting to inject an encryption key into NICI. 
	public static final int NE_XFLM_NICI_FIND_INIT						= 0x8105710E;			// Error occurred while attempting to initialize NICI to find information on a NICI encryption key.
	public static final int NE_XFLM_NICI_FIND_OBJECT					= 0x8105710F;			// Error occurred while attempting to find information on a NICI encryption key.
	public static final int NE_XFLM_NICI_KEY_NOT_FOUND					= 0x81057110;			// Could not find the NICI encryption key or information on the NICI encryption key.
	public static final int NE_XFLM_NICI_ENC_INIT_FAILED				= 0x81057111;			// Error occurred while initializing NICI to encrypt data.
	public static final int NE_XFLM_NICI_ENCRYPT_FAILED				= 0x81057112;			// Error occurred while encrypting data.
	public static final int NE_XFLM_NICI_DECRYPT_INIT_FAILED			= 0x81057113;			// Error occurred while initializing NICI to decrypt data.
	public static final int NE_XFLM_NICI_DECRYPT_FAILED				= 0x81057114;			// Error occurred while decrypting data.
	public static final int NE_XFLM_NICI_WRAPKEY_NOT_FOUND			= 0x81057115;			// Could not find the NICI encryption key used to wrap another NICI encryption key.
	public static final int NE_XFLM_NOT_EXPECTING_PASSWORD			= 0x81057116;			// Password supplied when none was expected.
	public static final int NE_XFLM_EXPECTING_PASSWORD					= 0x81057117;			// No password supplied when one was required.
	public static final int NE_XFLM_EXTRACT_KEY_FAILED					= 0x81057118;			// Error occurred while attempting to extract a NICI encryption key.
	public static final int NE_XFLM_NICI_INIT_FAILED					= 0x81057119;			// Error occurred while initializing NICI.
	public static final int NE_XFLM_BAD_ENCKEY_SIZE						= 0x8105711A;			// Bad encryption key size found in roll-forward log packet.
	public static final int NE_XFLM_ENCRYPTION_UNAVAILABLE			= 0x8105711B;			// Attempt was made to encrypt data when NICI is unavailable.
	public static final int NE_XFLM_LAST_NICI_ERROR						= 0x8105711C;			// NOTE: This is not an error code - do not document
}
