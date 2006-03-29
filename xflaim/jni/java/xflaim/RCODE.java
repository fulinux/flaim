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
// $Id: RCODE.java 3110 2006-01-19 13:09:08 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

package xflaim;

/**
 * Provides enums for some of the error codes that XFlaim might return in an
 * {@link xflaim.XFlaimException XFlaimException}.
 */

public final class RCODE
{
	public static final int NE_XFLM_OK 										= 0;

	public static final int NE_XFLM_FIRST_GENERAL_ERROR				= 0x81050000;			// Used for checking error boundaries
	public static final int NE_XFLM_BOF_HIT								= 0x81050001;			// Beginning of file hit
	public static final int NE_XFLM_EOF_HIT								= 0x81050002;			// End of file hit
	public static final int NE_XFLM_END										= 0x81050003;			// End of file for gedcom routine.  This is an internal error
	public static final int NE_XFLM_EXISTS									= 0x81050004;			// Record already exists
	public static final int NE_XFLM_NOT_FOUND								= 0x81050005;			// A record, key, or key reference was not found
	public static final int NE_XFLM_BAD_PREFIX							= 0x81050006;			// Invalid prefix name or number
	public static final int NE_XFLM_ATTRIBUTE_PURGED					= 0x81050007;			// Attribute cannot be used - it is being purged.
	public static final int NE_XFLM_BAD_COLLECTION						= 0x81050008;			// Invalid Collection Number
	public static final int NE_XFLM_NO_ROOT_BLOCK						= 0x81050009;			// LFILE does not have a root block.
	public static final int NE_XFLM_ILLEGAL_DATA_COMPONENT			= 0x8105000A;			// Cannot use ELM_ROOT_TAG as a data component in an index.
	public static final int NE_XFLM_CANNOT_CHANGE_NAME_ID				= 0x8105000B;			// Cannot change the element name id or attribute name id
	public static final int NE_XFLM_BAD_DATA_TYPE						= 0x8105000C;			// Bad data type
	public static final int NE_XFLM_MUST_INDEX_ON_PRESENCE			= 0x8105000D;			// When using ELM_ROOT_TAG in an index component, must specify PRESENCE indexing only
	public static final int NE_XFLM_BAD_IX									= 0x8105000E;			// Invalid Index Number Given
	public static final int NE_XFLM_BACKUP_ACTIVE						= 0x8105000F;			// Operation could not be completed - a backup is being performed
	public static final int NE_XFLM_SERIAL_NUM_MISMATCH				= 0x81050010;			// Comparison of serial numbers failed
	public static final int NE_XFLM_BAD_RFL_DB_SERIAL_NUM				= 0x81050011;			// Bad database serial number in RFL file header
	public static final int NE_XFLM_BTREE_ERROR							= 0x81050012;			// The B-Tree in the file system is bad
	public static final int NE_XFLM_BTREE_FULL							= 0x81050013;			// The B-tree in the file system is full
	public static final int NE_XFLM_BAD_RFL_FILE_NUMBER				= 0x81050014;			// Bad RFL file number in RFL file header
	public static final int NE_XFLM_CANNOT_DEL_ELEMENT					= 0x81050015;			// Cannot delete an element definition because it is in use
	public static final int NE_XFLM_CANNOT_MOD_DATA_TYPE				= 0x81050016;			// Cannot modify an element or attribute's data type
	public static final int NE_XFLM_CONV_BAD_DEST_TYPE					= 0x81050017;			// Bad destination type specified for conversion
	public static final int NE_XFLM_CONV_BAD_DIGIT						= 0x81050018;			// Non-numeric digit found in text to numeric conversion
	public static final int NE_XFLM_CONV_BAD_SRC_TYPE					= 0x81050019;			// Bad source type specified for conversion
	public static final int NE_XFLM_CONV_DEST_OVERFLOW					= 0x8105001A;			// Destination buffer not large enough to hold converted data
	public static final int NE_XFLM_CONV_ILLEGAL							= 0x8105001B;			// Illegal conversion -- not supported
	public static final int NE_XFLM_CONV_NULL_SRC						= 0x8105001C;			// Source cannot be a NULL pointer in conversion
	public static final int NE_XFLM_CONV_NULL_DEST						= 0x8105001D;			// Destination cannot be a NULL pointer in conversion
	public static final int NE_XFLM_CONV_NUM_OVERFLOW					= 0x8105001E;			// Numeric overflow (GT upper bound
	public static final int NE_XFLM_CONV_NUM_UNDERFLOW					= 0x8105001F;			// Numeric underflow (LT lower bound
	public static final int NE_XFLM_BAD_ELEMENT_NUM						= 0x81050020;			// Bad element number
	public static final int NE_XFLM_BAD_ATTRIBUTE_NUM					= 0x81050021;			// Bad attribute number
	public static final int NE_XFLM_DATA_ERROR							= 0x81050022;			// Data in the database is invalid
	public static final int NE_XFLM_DB_HANDLE								= 0x81050023;			// Out of FLAIM Session Database handles
	public static final int NE_XFLM_INVALID_FILE_SEQUENCE				= 0x81050024;			// Inc. backup file provided during a restore is invalid
	public static final int NE_XFLM_ILLEGAL_OP							= 0x81050025;			// Illegal operation for database
	public static final int NE_XFLM_DUPLICATE_ELEMENT_NUM				= 0x81050026;			// Element definition number already used
	public static final int NE_XFLM_CANNOT_CONVERT						= 0x81050027;			// Condition occurred which prevents database conversion
	public static final int NE_XFLM_UNSUPPORTED_VERSION				= 0x81050028;			// Db version is an unsupported ver of FLAIM (ver 1.2
	public static final int NE_XFLM_ILLEGAL_TRANS						= 0x81050029;			// Attempt to start an illegal type of transaction
	public static final int NE_XFLM_ILLEGAL_TRANS_OP					= 0x8105002A;			// Illegal operation for transaction type
	public static final int NE_XFLM_INCOMPLETE_LOG						= 0x8105002B;			// Incomplete log record encountered during recovery
	public static final int NE_XFLM_ILLEGAL_INDEX_DEF					= 0x8105002C;			// Index definition is bad
	public static final int NE_XFLM_ILLEGAL_INDEX_ON					= 0x8105002D;			// The "IndexOn" attribute of an index definition has an illegal value
	public static final int NE_XFLM_ILLEGAL_INDEX_ELM_PATH_DEF		= 0x8105002E;			// The ElementPath element definition is illegal.
	public static final int NE_XFLM_KEY_NOT_FOUND						= 0x8105002F;			// A key|reference is not found -- modify/delete error
	public static final int NE_XFLM_MEM										= 0x81050030;			// General memory allocation error
	public static final int NE_XFLM_BAD_RFL_SERIAL_NUM					= 0x81050031;			// Bad serial number in RFL file header
	public static final int NE_XFLM_NEWER_FLAIM							= 0x81050032;			// Running old code on a newer database code must be upgraded
	public static final int NE_XFLM_CANNOT_MOD_ELEMENT_STATE			= 0x81050033;			// Attempted to change an element state illegally
	public static final int NE_XFLM_CANNOT_MOD_ATTRIBUTE_STATE		= 0x81050034;			// Attempted to change an attribute state illegally
	public static final int NE_XFLM_NO_MORE_ELEMENT_NUMS				= 0x81050035;			// The highest element number has already been used, cannot create more
	public static final int NE_XFLM_NO_TRANS_ACTIVE						= 0x81050036;			// Attempted to updated DB outside transaction
	public static final int NE_XFLM_NOT_UNIQUE							= 0x81050037;			// Found Duplicate key for unique index
	public static final int NE_XFLM_NOT_FLAIM								= 0x81050038;			// Opened a file that was not a FLAIM file
	public static final int NE_XFLM_NO_HTTP_STACK						= 0x81050039;			// No http stack was loaded
	public static final int NE_XFLM_OLD_VIEW								= 0x8105003A;			// While reading was unable to get previous version of block
	public static final int NE_XFLM_SHARED_LOCK							= 0x8105003B;			// A transaction cannot be started within a shared lock
	public static final int NE_XFLM_SYNTAX									= 0x8105003C;			// Dictionary record has improper syntax
	public static final int NE_XFLM_CALLBACK_FAILURE					= 0x8105003D;			// Callback failure
	public static final int NE_XFLM_TRANS_ACTIVE							= 0x8105003E;			// Attempted to close DB while transaction was active
	public static final int NE_XFLM_RFL_TRANS_GAP						= 0x8105003F;			// A gap was found in the transaction sequence in the RFL
	public static final int NE_XFLM_BAD_COLLATED_KEY					= 0x81050040;			// Something in collated key is bad.
	public static final int NE_XFLM_UNSUPPORTED_FEATURE				= 0x81050041;			// Attempting a feature that is not supported for the database version.
	public static final int NE_XFLM_MUST_DELETE_INDEXES				= 0x81050042;			// Attempting to delete a container that has indexes defined for it.  Indexes must be deleted first.
	public static final int NE_XFLM_RFL_INCOMPLETE						= 0x81050043;			// RFL file is incomplete.
	public static final int NE_XFLM_CANNOT_RESTORE_RFL_FILES			= 0x81050044;			// Cannot restore RFL files - not using multiple RFL files.
	public static final int NE_XFLM_INCONSISTENT_BACKUP				= 0x81050045;			// A problem (corruption, etc.
	public static final int NE_XFLM_BLOCK_CRC								= 0x81050046;			// Block CRC error
	public static final int NE_XFLM_ABORT_TRANS							= 0x81050047;			// Attempted operation after a critical error - should abort transaction
	public static final int NE_XFLM_NOT_RFL								= 0x81050048;			// Attempted to open RFL file which was not an RFL file
	public static final int NE_XFLM_BAD_RFL_PACKET						= 0x81050049;			// RFL packet was bad
	public static final int NE_XFLM_DATA_PATH_MISMATCH					= 0x8105004A;			// Bad data path specified to open database
	public static final int NE_XFLM_HTTP_REGISTER_FAILURE				= 0x8105004B;			// Unable to register HTTP functions
	public static final int NE_XFLM_HTTP_DEREG_FAILURE					= 0x8105004C;			// Unable to deregister HTTP functions
	public static final int NE_XFLM_IX_FAILURE							= 0x8105004D;			// Indexing process failed, non-unique data was found when a unique index was being created
	public static final int NE_XFLM_HTTP_SYMS_EXIST						= 0x8105004E;			// Tried to import new http related symbols before unimporting the old ones
	public static final int NE_XFLM_DB_ALREADY_REBUILT					= 0x8105004F;			// Database has already been rebuilt
	public static final int NE_XFLM_FILE_EXISTS							= 0x81050050;			// Attempt to create a database or store file, but the file already exists
	public static final int NE_XFLM_SYM_RESOLVE_FAIL					= 0x81050051;			// Call to SAL_ModResolveSym failed.
	public static final int NE_XFLM_BAD_SERVER_CONNECTION				= 0x81050052;			// Connection to FLAIM server is bad
	public static final int NE_XFLM_CLOSING_DATABASE					= 0x81050053;			// Database is being closed due to a critical erro
	public static final int NE_XFLM_INVALID_CRC							= 0x81050054;			// CRC could not be verified.
	public static final int NE_XFLM_BAD_UTF8								= 0x81050055;			// An invalid byte sequence was found in a UTF-8 string
	public static final int NE_XFLM_NOT_IMPLEMENTED						= 0x81050056;			// function not implemented (possibly client/server
	public static final int NE_XFLM_MUTEX_OPERATION_FAILED			= 0x81050057;			// Mutex operation failed
	public static final int NE_XFLM_MUTEX_UNABLE_TO_LOCK				= 0x81050058;			// Unable to get the mutex lock
	public static final int NE_XFLM_SEM_OPERATION_FAILED				= 0x81050059;			// Semaphore operation failed
	public static final int NE_XFLM_SEM_UNABLE_TO_LOCK					= 0x8105005A;			// Unable to get the semaphore lock
	public static final int NE_XFLM_BAD_PLATFORM_FORMAT				= 0x8105005B;			// Cannot support platform format
	public static final int NE_XFLM_HDR_CRC								= 0x8105005C;			// Header has a bad CRC.
	public static final int NE_XFLM_NO_NAME_TABLE						= 0x8105005D;			// No name table was set up for the database
	public static final int NE_XFLM_MULTIPLE_MATCHES					= 0x8105005E;			// Multiple entries match the passed in name in the name table
	public static final int NE_XFLM_BAD_REFERENCE						= 0x8105005F;			// Bad reference in the dictionary
	public static final int NE_XFLM_INCONSISTENT_BTREE					= 0x81050060;			// F_BTree::FlmBtCheck(
	public static final int NE_XFLM_KEY_TRUNC								= 0x81050061;			// FlmBtFind truncated the return key.  Buffer was too small
	public static final int NE_XFLM_UNALLOWED_UPGRADE					= 0x81050062;			// FlmDbUpgrade cannot upgrade the database
	public static final int NE_XFLM_BTREE_BAD_STATE						= 0x81050063;			// Btree function called before proper setup steps taken
	public static final int NE_XFLM_DUPLICATE_ATTRIBUTE_NUM			= 0x81050064;			// Attribute definition number already used
	public static final int NE_XFLM_BAD_QUERY_SOURCE					= 0x81050065;			// No query source, or bad source
	public static final int NE_XFLM_DUPLICATE_INDEX_NUM				= 0x81050066;			// Index definition number already used
	public static final int NE_XFLM_DUPLICATE_COLLECTION_NUM			= 0x81050067;			// Collection definition number already used
	public static final int NE_XFLM_DUPLICATE_ELEMENT_NAME			= 0x81050068;			// Element definition name already used
	public static final int NE_XFLM_DUPLICATE_ATTRIBUTE_NAME			= 0x81050069;			// Attribute definition name already used
	public static final int NE_XFLM_DUPLICATE_INDEX_NAME				= 0x8105006A;			// Index definition name already used
	public static final int NE_XFLM_DUPLICATE_COLLECTION_NAME		= 0x8105006B;			// Collection definition name already used
	public static final int NE_XFLM_ELEMENT_PURGED						= 0x8105006C;			// Element cannot be used - it is being purged.
	public static final int NE_XFLM_TOO_MANY_OPEN_FILES				= 0x8105006D;			// No session file handles could be closed - in to increase via FlmSetMaxPhysOpens(
	public static final int NE_XFLM_DATABASE_OPEN						= 0x8105006E;			// Access Denied from setting in log header
	public static final int NE_XFLM_CACHE_ERROR							= 0x8105006F;			// Cache Block is somehow corrupt
	public static final int NE_XFLM_BTREE_KEY_SIZE						= 0x81050070;			// Btree key passed in is too large, i.e. > 640.
	public static final int NE_XFLM_BLOB_MISSING_FILE					= 0x81050071;			// Missing BLOB file on add/modify
	public static final int NE_XFLM_DB_FULL								= 0x81050072;			// Database is full, cannot create more blocks
	public static final int NE_XFLM_TIMEOUT								= 0x81050073;			// Query operation timed out
	public static final int NE_XFLM_CURSOR_SYNTAX						= 0x81050074;			// Cursor operation had improper syntax
	public static final int NE_XFLM_THREAD_ERR							= 0x81050075;			// Thread Error
	public static final int NE_XFLM_EMPTY_QUERY							= 0x81050076;			// Warning: Query has no results
	public static final int NE_XFLM_INDEX_OFFLINE						= 0x81050077;			// Warning: Index is offline and being rebuild
	public static final int NE_XFLM_TRUNCATED_KEY						= 0x81050078;			// Warning: Can't evaluate truncated key against selection criteria
	public static final int NE_XFLM_INVALID_PARM							= 0x81050079;			// Invalid parm
	public static final int NE_XFLM_USER_ABORT							= 0x8105007A;			// User or application aborted the operation
	public static final int NE_XFLM_RFL_DEVICE_FULL						= 0x8105007B;			// No space on RFL device for logging
	public static final int NE_XFLM_MUST_WAIT_CHECKPOINT				= 0x8105007C;			// Must wait for a checkpoint before starting transaction - due to disk problems - usually in RFL volume.
	public static final int NE_XFLM_NAMED_SEMAPHORE_ERR				= 0x8105007D;			// Something bad happened with the named semaphore class (F_NamedSemaphore
	public static final int NE_XFLM_LOAD_LIBRARY							= 0x8105007E;			// Failed to load a shared library module
	public static final int NE_XFLM_UNLOAD_LIBRARY						= 0x8105007F;			// Failed to unload a shared library module
	public static final int NE_XFLM_IMPORT_SYMBOL						= 0x81050080;			// Failed to import a symbol from a shared library module
	public static final int NE_XFLM_ILLEGAL_DATA_TYPE					= 0x81050081;			// Data type specified in "type" attribute is illegal
	public static final int NE_XFLM_ILLEGAL_STATE						= 0x81050082;			// State specified in "state" attribute is illegal
	public static final int NE_XFLM_ILLEGAL_ELEMENT_NAME				= 0x81050083;			// Element name specified is illegal
	public static final int NE_XFLM_ILLEGAL_ATTRIBUTE_NAME			= 0x81050084;			// Attribute name specified is illegal
	public static final int NE_XFLM_ILLEGAL_COLLECTION_NAME			= 0x81050085;			// Collection name specified is illegal
	public static final int NE_XFLM_ILLEGAL_INDEX_NAME					= 0x81050086;			// Index name specified is illegal
	public static final int NE_XFLM_ILLEGAL_ELEMENT_NUMBER			= 0x81050087;			// Element number specified is illegal
	public static final int NE_XFLM_ILLEGAL_ATTRIBUTE_NUMBER			= 0x81050088;			// Attribute number specified is illegal
	public static final int NE_XFLM_ILLEGAL_COLLECTION_NUMBER		= 0x81050089;			// Collection number specified is illegal
	public static final int NE_XFLM_ILLEGAL_INDEX_NUMBER				= 0x8105008A;			// Index number specified is illegal
	public static final int NE_XFLM_COLLECTION_NAME_MISMATCH			= 0x8105008B;			// Collection name and number do not correspond to each other
	public static final int NE_XFLM_ELEMENT_NAME_MISMATCH				= 0x8105008C;			// Element name and number do not correspond to each other
	public static final int NE_XFLM_ATTRIBUTE_NAME_MISMATCH			= 0x8105008D;			// Attribute name and number do not correspond to each other
	public static final int NE_XFLM_BAD_INDEX_DEF_SYNTAX				= 0x8105008E;			// Bad index definition syntax
	public static final int NE_XFLM_DUPLICATE_KEY_COMPONENT			= 0x8105008F;			// Duplicate key component specified in index definition
	public static final int NE_XFLM_DUPLICATE_DATA_COMPONENT			= 0x81050090;			// Duplicate data component specified in index definition
	public static final int NE_XFLM_MISSING_KEY_COMPONENT				= 0x81050091;			// Index definition is missing a key component
	public static final int NE_XFLM_MISSING_DATA_COMPONENT			= 0x81050092;			// Index definition is missing a data component
	public static final int NE_XFLM_INVALID_INDEX_OPTION				= 0x81050093;			// Invalid index option specified on index definition
	public static final int NE_XFLM_DICT_TYPE_MISMATCH					= 0x81050094;			// Dictionary type attribute does not match root element tag for dictionary definition
	public static final int NE_XFLM_NO_MORE_ATTRIBUTE_NUMS			= 0x81050095;			// The highest attribute number has already been used, cannot create more
	public static final int NE_XFLM_MISSING_ELEMENT_NAME				= 0x81050096;			// Missing element name from element definition
	public static final int NE_XFLM_MISSING_ATTRIBUTE_NAME			= 0x81050097;			// Missing attribute name from attribute definition
	public static final int NE_XFLM_MISSING_ELEMENT_NUMBER			= 0x81050098;			// Missing element number from element definition
	public static final int NE_XFLM_MISSING_ATTRIBUTE_NUMBER			= 0x81050099;			// Missing attribute number from attribute definition
	public static final int NE_XFLM_MISSING_INDEX_NAME					= 0x8105009A;			// Missing index name from index definition
	public static final int NE_XFLM_MISSING_INDEX_NUMBER				= 0x8105009B;			// Missing index number from index definition
	public static final int NE_XFLM_MISSING_COLLECTION_NAME			= 0x8105009C;			// Missing collection name from collection definition
	public static final int NE_XFLM_MISSING_COLLECTION_NUMBER		= 0x8105009D;			// Missing collection number from collection definition
	public static final int NE_XFLM_BAD_SEN								= 0x8105009E;			// Invalid SEN value
	public static final int NE_XFLM_MISSING_DICT_NAME					= 0x8105009F;			// Missing dictionary name from definition
	public static final int NE_XFLM_NO_MORE_INDEX_NUMS					= 0x810500A0;			// The highest index number has already been used, cannot create more
	public static final int NE_XFLM_NO_MORE_COLLECTION_NUMS			= 0x810500A1;			// The highest collection number has already been used, cannot create more
	public static final int NE_XFLM_CANNOT_DEL_ATTRIBUTE				= 0x810500A2;			// Cannot delete an attribute definition because it is in use
	public static final int NE_XFLM_TOO_MANY_PENDING_DOCUMENTS		= 0x810500A3;			// Too many documents in the pending document list.
	public static final int NE_XFLM_UNSUPPORTED_INTERFACE				= 0x810500A4;			// Asked a COM object for an interface that it doesn't support
	public static final int NE_XFLM_UNSUPPORTED_CLASS					= 0x810500A5;			// Asked for a COM object that doesn't exist
	public static final int NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG			= 0x810500A6;			// ELM_ROOT_TAG, if used, must be the root component of an index definition
	public static final int NE_XFLM_DUP_SIBLING_IX_COMPONENTS		= 0x810500A7;			// Sibling components in an index definition cannot have the same element or attribute number
	public static final int NE_XFLM_RFL_FILE_NOT_FOUND					= 0x810500A8;			// Could not open an RFL file.
	public static final int NE_XFLM_BAD_RCODE_TABLE						= 0x810500A9;			// The error code tables are incorrect
	public static final int NE_XFLM_FAILURE								= 0x810500AA;			// Internal failure
	public static final int NE_XFLM_ILLEGAL_KEY_COMPONENT_NUM		= 0x810500AB;			// Key component of zero in index definition is not allowed
	public static final int NE_XFLM_ILLEGAL_DATA_COMPONENT_NUM		= 0x810500AC;			// Data component of zero in index definition is not allowed
	public static final int NE_XFLM_CLASSNOTAVAILABLE					= 0x810500AD;			// PSA subsystem asked for a COM server we don't implement
	public static final int NE_XFLM_BUFFER_OVERFLOW						= 0x810500AE;			// Buffer overflow
	public static final int NE_XFLM_ILLEGAL_PREFIX_NUMBER				= 0x810500AF;			// Prefix number is out of range
	public static final int NE_XFLM_MISSING_PREFIX_NAME				= 0x810500B0;			// Prefix Name is missing in definition
	public static final int NE_XFLM_MISSING_PREFIX_NUMBER				= 0x810500B1;			// Prefix number is missing in definition
	public static final int NE_XFLM_INVALID_ELEMENT_NAME				= 0x810500B2;			// Specified Element name is not present in Db.
	public static final int NE_XFLM_INVALID_ATTRIBUTE_NAME			= 0x810500B3;			// Specified Attribute name is not present in Db.
	public static final int NE_XFLM_DUPLICATE_PREFIX_NAME				= 0x810500B4;			// Prefix definition name already used
	public static final int NE_XFLM_KEY_OVERFLOW							= 0x810500B5;			// Generated key too large
	public static final int NE_XFLM_LAST_GENERAL_ERROR					= 0x810500B6;			// Used for checking error boundaries
	public static final int NE_XFLM_FIRST_DOM_ERROR						= 0x81051000;			// Used for checking error boundaries
	public static final int NE_XFLM_DOM_INDEX_SIZE_ERR					= 0x81051001;			// Index or size is negative or greater than the allowed value
	public static final int NE_XFLM_DOM_DOMSTRING_SIZE_ERR			= 0x81051002;			// The specified range of text does not fit into a string
	public static final int NE_XFLM_DOM_HIERARCHY_REQUEST_ERR		= 0x81051003;			// Attempt to insert a node somewhere it doesn't belong
	public static final int NE_XFLM_DOM_WRONG_DOCUMENT_ERR			= 0x81051004;			// A node is being used in a different document than the one that created it
	public static final int NE_XFLM_DOM_INVALID_CHARACTER_ERR		= 0x81051005;			// An invalid character was specified (in a name, etc.
	public static final int NE_XFLM_DOM_NO_DATA_ALLOWED_ERR			= 0x81051006;			// Data is specified for a node that does not support data
	public static final int NE_XFLM_DOM_NO_MOD_ALLOWED_ERR			= 0x81051007;			// Attempt to modify a read-only object
	public static final int NE_XFLM_DOM_NOT_FOUND_ERR					= 0x81051008;			// An attempt was made to reference a node in a context where it doesn't exist
	public static final int NE_XFLM_DOM_NOT_SUPPORTED_ERR				= 0x81051009;			// The implementation does not support the object or operation
	public static final int NE_XFLM_DOM_INUSE_ATTRIBUTE_ERR			= 0x8105100A;			// Attempt to add an attribute that is already in use elsewhere
	public static final int NE_XFLM_DOM_INVALID_STATE_ERR				= 0x8105100B;			// Attempt to use an object that is no longer usable
	public static final int NE_XFLM_DOM_SYNTAX_ERR						= 0x8105100C;			// Invalid or illegal string specified
	public static final int NE_XFLM_DOM_INVALID_MOD_ERR				= 0x8105100D;			// Attempt to modify the type of an underlying object
	public static final int NE_XFLM_DOM_NAMESPACE_ERR					= 0x8105100E;			// Attempt to create or change an object in a way which is incorrect with regard to namespaces
	public static final int NE_XFLM_DOM_INVALID_ACCESS_ERR			= 0x8105100F;			// A parameter or operation is not supported by the underlying object
	public static final int NE_XFLM_DOM_DATA_ERROR						= 0x81051010;			// The document is corrupt
	public static final int NE_XFLM_DOM_NODE_NOT_FOUND					= 0x81051011;			// Could not access the DOM node specified.
	public static final int NE_XFLM_DOM_NODE_OUT_OF_DATE				= 0x81051012;			// F_DOMNode object is out of date, cannot use.
	public static final int NE_XFLM_DOM_INVALID_CHILD_TYPE			= 0x81051013;			// Cannot insert a child of the specified type
	public static final int NE_XFLM_LAST_DOM_ERROR						= 0x81051014;			// Used for checking error boundaries
	public static final int NE_XFLM_FIRST_IO_ERROR						= 0x81052000;			// Used for checking error boundaries
	public static final int NE_XFLM_IO_ACCESS_DENIED					= 0x81052001;			// Access denied. Caller is not allowed access to a file.
	public static final int NE_XFLM_IO_BAD_FILE_HANDLE					= 0x81052002;			// Bad file handle
	public static final int NE_XFLM_IO_COPY_ERR							= 0x81052003;			// Copy error
	public static final int NE_XFLM_IO_DISK_FULL							= 0x81052004;			// Disk full
	public static final int NE_XFLM_IO_END_OF_FILE						= 0x81052005;			// End of file
	public static final int NE_XFLM_IO_OPEN_ERR							= 0x81052006;			// Error opening file
	public static final int NE_XFLM_IO_SEEK_ERR							= 0x81052007;			// File seek error
	public static final int NE_XFLM_IO_MODIFY_ERR						= 0x81052008;			// File modify error
	public static final int NE_XFLM_IO_PATH_NOT_FOUND					= 0x81052009;			// Path not found
	public static final int NE_XFLM_IO_TOO_MANY_OPEN_FILES			= 0x8105200A;			// Too many files open
	public static final int NE_XFLM_IO_PATH_TOO_LONG					= 0x8105200B;			// Path too long
	public static final int NE_XFLM_IO_NO_MORE_FILES					= 0x8105200C;			// No more files in directory
	public static final int NE_XFLM_IO_DELETING_FILE					= 0x8105200D;			// Had error deleting a file
	public static final int NE_XFLM_IO_FILE_LOCK_ERR					= 0x8105200E;			// File lock error
	public static final int NE_XFLM_IO_FILE_UNLOCK_ERR					= 0x8105200F;			// File unlock error
	public static final int NE_XFLM_IO_PATH_CREATE_FAILURE			= 0x81052010;			// Path create failed
	public static final int NE_XFLM_IO_RENAME_FAILURE					= 0x81052011;			// File rename failed
	public static final int NE_XFLM_IO_INVALID_PASSWORD				= 0x81052012;			// Invalid file password
	public static final int NE_XFLM_SETTING_UP_FOR_READ				= 0x81052013;			// Had error setting up to do a read
	public static final int NE_XFLM_SETTING_UP_FOR_WRITE				= 0x81052014;			// Had error setting up to do a write
	public static final int NE_XFLM_IO_AT_PATH_ROOT						= 0x81052015;			// Currently positioned at the path root level
	public static final int NE_XFLM_INITIALIZING_IO_SYSTEM			= 0x81052016;			// Had error initializing the file system
	public static final int NE_XFLM_FLUSHING_FILE						= 0x81052017;			// Had error flushing a file
	public static final int NE_XFLM_IO_INVALID_PATH						= 0x81052018;			// Invalid path
	public static final int NE_XFLM_IO_CONNECT_ERROR					= 0x81052019;			// Failed to connect to a remote network resource
	public static final int NE_XFLM_OPENING_FILE							= 0x8105201A;			// Had error opening a file
	public static final int NE_XFLM_DIRECT_OPENING_FILE				= 0x8105201B;			// Had error opening a file for direct I/O
	public static final int NE_XFLM_CREATING_FILE						= 0x8105201C;			// Had error creating a file
	public static final int NE_XFLM_DIRECT_CREATING_FILE				= 0x8105201D;			// Had error creating a file for direct I/O
	public static final int NE_XFLM_READING_FILE							= 0x8105201E;			// Had error reading a file
	public static final int NE_XFLM_DIRECT_READING_FILE				= 0x8105201F;			// Had error reading a file using direct I/O
	public static final int NE_XFLM_WRITING_FILE							= 0x81052020;			// Had error writing to a file
	public static final int NE_XFLM_DIRECT_WRITING_FILE				= 0x81052021;			// Had error writing to a file using direct I/O
	public static final int NE_XFLM_POSITIONING_IN_FILE				= 0x81052022;			// Had error positioning within a file
	public static final int NE_XFLM_GETTING_FILE_SIZE					= 0x81052023;			// Had error getting file size
	public static final int NE_XFLM_TRUNCATING_FILE						= 0x81052024;			// Had error truncating a file
	public static final int NE_XFLM_PARSING_FILE_NAME					= 0x81052025;			// Had error parsing a file name
	public static final int NE_XFLM_CLOSING_FILE							= 0x81052026;			// Had error closing a file
	public static final int NE_XFLM_GETTING_FILE_INFO					= 0x81052027;			// Had error getting file information
	public static final int NE_XFLM_EXPANDING_FILE						= 0x81052028;			// Had error expanding a file (using direct I/O
	public static final int NE_XFLM_GETTING_FREE_BLOCKS				= 0x81052029;			// Had error getting free blocks from file system
	public static final int NE_XFLM_CHECKING_FILE_EXISTENCE			= 0x8105202A;			// Had error checking if a file exists
	public static final int NE_XFLM_RENAMING_FILE						= 0x8105202B;			// Had error renaming a file
	public static final int NE_XFLM_SETTING_FILE_INFO					= 0x8105202C;			// Had error setting file information
	public static final int NE_XFLM_LAST_IO_ERROR						= 0x8105202D;			// Used for checking error boundaries
	public static final int NE_XFLM_FIRST_NET_ERROR						= 0x81053000;			// Used for checking error boundaries
	public static final int NE_XFLM_SVR_NOIP_ADDR						= 0x81053001;			// IP address not found
	public static final int NE_XFLM_SVR_SOCK_FAIL						= 0x81053002;			// IP socket failure
	public static final int NE_XFLM_SVR_CONNECT_FAIL					= 0x81053003;			// TCP/IP connection failure
	public static final int NE_XFLM_SVR_BIND_FAIL						= 0x81053004;			// The TCP/IP services on your system may not be configured or installed.  If this POA is not to run Client/Server, use the /notcpip startup switch or disable TCP/IP through the NWADMIN snapin
	public static final int NE_XFLM_SVR_LISTEN_FAIL						= 0x81053005;			// TCP/IP listen failed
	public static final int NE_XFLM_SVR_ACCEPT_FAIL						= 0x81053006;			// TCP/IP accept failed
	public static final int NE_XFLM_SVR_SELECT_ERR						= 0x81053007;			// TCP/IP select failed
	public static final int NE_XFLM_SVR_SOCKOPT_FAIL					= 0x81053008;			// TCP/IP socket operation failed
	public static final int NE_XFLM_SVR_DISCONNECT						= 0x81053009;			// TCP/IP disconnected
	public static final int NE_XFLM_SVR_READ_FAIL						= 0x8105300A;			// TCP/IP read failed
	public static final int NE_XFLM_SVR_WRT_FAIL							= 0x8105300B;			// TCP/IP write failed
	public static final int NE_XFLM_SVR_READ_TIMEOUT					= 0x8105300C;			// TCP/IP read timeout
	public static final int NE_XFLM_SVR_WRT_TIMEOUT						= 0x8105300D;			// TCP/IP write timeout
	public static final int NE_XFLM_SVR_ALREADY_CLOSED					= 0x8105300E;			// Connection already closed
	public static final int NE_XFLM_LAST_NET_ERROR						= 0x8105300F;			// Used for checking error boundaries
	public static final int NE_XFLM_FIRST_QUERY_ERROR					= 0x81054000;			// Used for checking error boundaries
	public static final int NE_XFLM_Q_UNMATCHED_RPAREN					= 0x81054001;			// Unmatched right paren
	public static final int NE_XFLM_Q_UNEXPECTED_LPAREN				= 0x81054002;			// Unexpected left paren
	public static final int NE_XFLM_Q_UNEXPECTED_RPAREN				= 0x81054003;			// Unexpected left paren
	public static final int NE_XFLM_Q_EXPECTING_OPERAND				= 0x81054004;			// Expecting an operand
	public static final int NE_XFLM_Q_EXPECTING_OPERATOR				= 0x81054005;			// Expecting an operator
	public static final int NE_XFLM_Q_UNEXPECTED_COMMA					= 0x81054006;			// Unexpected comma
	public static final int NE_XFLM_Q_EXPECTING_LPAREN					= 0x81054007;			// Expecting a left paren
	public static final int NE_XFLM_Q_UNEXPECTED_VALUE					= 0x81054008;			// Unexpected value
	public static final int NE_XFLM_Q_INVALID_NUM_FUNC_ARGS			= 0x81054009;			// Invalid number of arguments for a function
	public static final int NE_XFLM_Q_UNEXPECTED_XPATH					= 0x8105400A;			// Unexpected XPATH
	public static final int NE_XFLM_Q_ILLEGAL_LBRACKET					= 0x8105400B;			// Illegal left bracket ([
	public static final int NE_XFLM_Q_ILLEGAL_RBRACKET					= 0x8105400C;			// Illegal right bracket (]
	public static final int NE_XFLM_Q_ILLEGAL_OPERAND					= 0x8105400D;			// Operand for some operator is not valid for that operator type
	public static final int NE_XFLM_Q_ALREADY_OPTIMIZED				= 0x8105400E;			// Cannot change certain things, query is already optimized
	public static final int NE_XFLM_Q_MISMATCHED_DB						= 0x8105400F;			// Database handle passed in does not match database associated with query.
	public static final int NE_XFLM_Q_ILLEGAL_OPERATOR					= 0x81054010;			// Illegal operator - cannot pass to query addOperator method
	public static final int NE_XFLM_Q_ILLEGAL_COMPARE_RULES			= 0x81054011;			// Illegal combination of comparison rules passed to addOperator method
	public static final int NE_XFLM_Q_CANNOT_CALL_METHOD				= 0x81054012;			// Attempt to call a method that an application is not allowed to call
	public static final int NE_XFLM_Q_INCOMPLETE_QUERY_EXPR			= 0x81054013;			// Query expression is incomplete
	public static final int NE_XFLM_Q_NOT_POSITIONED					= 0x81054014;			// Query not positioned due to previous error, cannot call getNext, getPrev, or getCurrent
	public static final int NE_XFLM_LAST_QUERY_ERROR					= 0x81054015;			// Used for checking error boundaries
}
